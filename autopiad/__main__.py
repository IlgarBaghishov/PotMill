import pandas as pd
import os, copy, time, pickle
from ase.io import write
from autopiad.tools import create_rcut_range, rcuts_to_string, nmaxes_to_string, lmaxes_to_string, twojmaxes_to_string
from autopiad.tools import ace_hyperparameters_to_string, snap_hyperparameters_to_string
from autopiad.tools import combined_ace_hyperparameters, combined_snap_hyperparameters, parse_inputfile, configparse
from autopiad.entropy.binary.optimizer import EntropyMaximizer 
from autopiad.featurize import featurize
from autopiad.vasp import vasp
from autopiad.lammps import lammps
from autopiad.fake_vasp import fake_vasp
from autopiad.fit import fit
from autopiad.pareto import pareto
import flux
import concurrent.futures
import flux.job
from executorlib import FluxJobExecutor, SingleNodeExecutor


def main():
    handle = flux.Flux()
    rs = flux.resource.status.ResourceStatusRPC(handle).get()
    rl = flux.resource.list.resource_list(handle).get()
    all_ncores = rl.all.ncores
    all_ngpus = rl.all.ngpus

    print("NODELIST:",rs.nodelist, " #CORES:",all_ncores, " #GPUS:",all_ngpus)

    start_path = os.getcwd()+'/'
    config = parse_inputfile(start_path+"inputfile")
    fitsnap_config = configparse(start_path + config['FitSNAP']['filename'])
    fitsnap_config = {section: dict(fitsnap_config.items(section)) for section in fitsnap_config.sections()}

    mlip = config["FitSNAP"]["mlip"]
    resume_mode = config["MODE"]["resume"]
    entropy_mode = config["MODE"]["entropy"]
    feature_mode = config["MODE"]["featurize"]
    vasp_mode = config["MODE"]["vasp"]
    fit_mode = config["MODE"]["fit"]
    pareto_mode = config["MODE"]["pareto"]
    fit_freq = config["MODE"]["fit_freq"]
    ncores_per_fit = config["MODE"]["ncores_per_fit"]
    auto_reduce_hps = config["MODE"]["auto_reduce_hyperparameters"]
    rcuts_list = create_rcut_range(config["RCUT"]["min_rcut"],config["RCUT"]["max_rcut"],config["RCUT"]["num_rcut"])
    if mlip == "ACE":
        hyperparameters_list = combined_ace_hyperparameters(config)
        hyperparameters_list_noeweight = combined_ace_hyperparameters(config, w_eweight=False)
        fitsnap_config["ACE"]["nmax"] = nmaxes_to_string(config["NMAX"]["max_nmax"])
        fitsnap_config["ACE"]["lmax"] = lmaxes_to_string(config["LMAX"]["max_lmax"])
    elif mlip == "SNAP":
        hyperparameters_list = combined_snap_hyperparameters(config)
        hyperparameters_list_noeweight = combined_snap_hyperparameters(config, w_eweight=False)
        fitsnap_config["BISPECTRUM"]["twojmax"] = twojmaxes_to_string(config["TWOJMAX"]["max_twojmax"])

    if not resume_mode and feature_mode:
        os.system("rm -rf "+start_path+"features")
        os.mkdir(start_path+"features")
    if not resume_mode and fit_mode:
        os.system("rm -rf "+start_path+"fits")
        os.mkdir(start_path+"fits")
    if not resume_mode and pareto_mode:
        os.system("rm -rf "+start_path+"costs")
        os.mkdir(start_path+"costs")
        os.system("rm -rf "+start_path+"pareto-front")
        os.mkdir(start_path+"pareto-front")
    if not resume_mode and vasp_mode:
        os.system("rm -rf "+start_path+"energy-configs")
        os.mkdir(start_path+"energy-configs")
        os.system("rm -rf "+start_path+"vasp-energy")
        os.mkdir(start_path+"vasp-energy")

    # if entropy_mode:
    #     em = EntropyMaximizer()
    # else:
    # scan the available configurations and sort them by size
    try:
        df = pd.read_hdf(start_path + config["DATA"]["data_path"]).iloc[:500,:]
    except:
        try:
            df = pd.read_pickle(start_path + config["DATA"]["data_path"], compression="gzip").iloc[:500,:]
            force_energy_filename = start_path + "force_energy.pkl"
            df.iloc[:,4:].to_pickle(force_energy_filename)
        except:
            raise
    index0 = 0
    index1 = df.shape[0]
    tasks = []
    first_index = [0]
    if not df.index.equals(pd.RangeIndex(0,df.shape[0],1)):
        df.reset_index(inplace=True)
    for i in range(index0,index1):
        atoms = df['ase_atoms'][i]
        n_atoms = len(atoms)
        tasks.append([i,n_atoms])
        first_index.append(first_index[-1]+1+3*n_atoms)
        if not os.path.isfile(start_path+"energy-configs/em_%i.dat"% i):
            write(start_path+"energy-configs/em_%i.dat"% i, atoms, format='vasp')
        if not os.path.isdir(start_path+"vasp-energy/vasp-em_%i"% i):
            os.makedirs(start_path+"vasp-energy/vasp-em_%i"% i)
    tasks.sort(key=lambda x: x[1]) #large systems are at the end, small systems are at the front

    in_process_featurizations = []
    in_process_tasks = []
    in_process_fits = []
    in_process_costs = []
    if resume_mode and os.path.isfile("checkpoint.pkl"):
        print("RESUMING FROM CHECKPOINT")
        with open("checkpoint.pkl", "rb") as f:
            (completed_featurizations, completed_tasks, completed_fits, completed_costs, job_ids_for_fit,
             feature_names, trigger_fit, wait_for_last_fit) = pickle.load(f)
            
        remaining_featurizations = [i for i in range(len(rcuts_list)) if i not in completed_featurizations]
        remaining_tasks = [task[0] for task in tasks if task[0] not in completed_tasks]
        remaining_fits = [i for i in range(len(hyperparameters_list)) if i not in completed_fits]
        remaining_costs = [i for i in range(len(hyperparameters_list_noeweight)) if i not in completed_costs]
        failed_tasks = []
    else:
        completed_featurizations = []
        completed_tasks = []
        completed_fits = []
        completed_costs = []
        remaining_featurizations = [i for i in range(len(rcuts_list))] if feature_mode else []
        remaining_tasks = [task[0] for task in tasks] if vasp_mode else []
        remaining_fits = [i for i in range(len(hyperparameters_list))] if fit_mode else []
        remaining_costs = [i for i in range(len(hyperparameters_list_noeweight))] if pareto_mode else []
        failed_tasks = []
        job_ids_for_fit = []
        feature_names = []
        trigger_fit = 0 if vasp_mode else 2
        wait_for_last_fit = 0

    print(len(remaining_tasks)," TASKS REMAINING  --- ", len(in_process_tasks)," TASKS IN PROCESS  --- ", len(completed_tasks), " COMPLETED TASKS")

    if vasp_mode: vasp_futures = set()
    if feature_mode: featurization_futures = set()
    if fit_mode: fitting_futures = set()
    if pareto_mode: cost_futures = set()

    start_time = time.time()
    with FluxJobExecutor(flux_executor_pmi_mode="pmi2", flux_log_files=True) as exe:
    # with SingleNodeExecutor() as exe:

        rl = flux.resource.list.resource_list(handle).get()
        print(rl.free.ncores, "CORES FREE ",all_ncores, "CORES TOTAL")
        print(rl.free.ngpus, "GPUS FREE ",all_ngpus, "GPUS TOTAL")
        ncores_per_featurization = (rl.free.ncores - 2*rl.free.ngpus)//len(rs.nodelist) - 1
        # ncores_per_featurization = 29
        print("Number of cores allocated for featurization step is", ncores_per_featurization)

        print("Featurization step...")
        for i in remaining_featurizations:
            rcuts = rcuts_list[i]
            feature_directory = start_path + "features/" + rcuts_to_string(rcuts, delimiter='_')
            if not os.path.isdir(feature_directory):
                os.mkdir(feature_directory)
            fs = exe.submit(featurize, df['ase_atoms'].to_list(), config, fitsnap_config, rcuts,
                            resource_dict={"cores": ncores_per_featurization, "gpus_per_core": 0,
                                           "num_nodes": 1, "cwd": feature_directory})
            fs.task_ = i
            featurization_futures.add(fs)
            in_process_featurizations.append(i)
        remaining_featurizations = []

        while True:
            
            if (len(remaining_featurizations) == 0 and len(in_process_featurizations) == 0) and \
            (len(remaining_tasks) == 0 and len(in_process_tasks) == 0) and \
            (len(remaining_fits) == 0 and len(in_process_fits) == 0) and wait_for_last_fit == 0:
                break

            rl = flux.resource.list.resource_list(handle).get()
            # if len(completed_tasks) == len(tasks) and len(remaining_fits) != 0:
            print("It has been %.3f seconds since the last check." % (time.time() - start_time))
            start_time = time.time()
            print(rl.free.ncores, "CORES FREE ", all_ncores, "CORES TOTAL", rl.free.ngpus, "GPUS FREE ", all_ngpus, "GPUS TOTAL")
            print(len(remaining_featurizations), len(in_process_featurizations), len(completed_featurizations), len(remaining_tasks),
                  len(in_process_tasks), len(completed_tasks), len(remaining_fits), len(in_process_fits), len(completed_fits),
                  wait_for_last_fit)


            # print("SCHEDULING VASP TASKS")
            if len(remaining_tasks)>0:
                rl = flux.resource.list.resource_list(handle).get()
                n_gpus_free = rl.free.ngpus
                n_cores_free = rl.free.ncores
            
                while n_gpus_free>=1 and len(remaining_tasks)>0 and len(in_process_tasks)<all_ngpus:
                # while n_gpus_free>=1 and len(remaining_tasks)>0 and len(in_process_tasks)<(all_ngpus-1):
                    
                    task = remaining_tasks.pop(0)
                    input_file = "energy-configs/em_%i.dat"%task
                    vasp_directory = start_path + "vasp-energy/vasp-em_%i/"%task

                    print("RUNNING ", task, "on GPUs", vasp_directory, input_file)
                    # fs = exe.submit(fake_vasp, force_energy_filename, task, first_index[task],
                    #                 resource_dict={"cores": 1, "gpus_per_core": 1, "num_nodes": 1, "cwd": vasp_directory})
                    fs = exe.submit(vasp, start_path, start_path+input_file, task, first_index[task],
                                    resource_dict={"cores": 1, "gpus_per_core": 0, "num_nodes": 1, "cwd": vasp_directory})
                    # fs = exe.submit(lammps, start_path, start_path+input_file, task, first_index[task],
                    #                 resource_dict={"cores": 1, "gpus_per_core": 0, "num_nodes": 1, "cwd": vasp_directory})
                    fs.task_ = task
                    vasp_futures.add(fs)
                    in_process_tasks.append(task)
                    n_gpus_free-=1


            # print("PROCESSING VASP FUTURES")
            if vasp_mode:
                print("nvasp_futures was ", len(vasp_futures), vasp_futures)
                vasp_done, vasp_futures = concurrent.futures.wait(vasp_futures, timeout=0.1)
                print("nvasp_futures is ", len(vasp_futures), vasp_futures)
                for fut in vasp_done:
                    completed_tasks.append(fut.task_)

                    if len(completed_tasks) == len(tasks) and len(in_process_fits) == 0 and len(in_process_featurizations) == 0:
                        trigger_fit = 1
                        print("Triggering last fit: ", len(completed_tasks))
                    elif len(completed_tasks) == len(tasks) and len(in_process_fits) != 0:
                        wait_for_last_fit = 1
                    elif len(completed_tasks)%fit_freq == 0 and len(in_process_fits) == 0 and len(in_process_featurizations) == 0:
                        trigger_fit = 1
                        print("Triggering fit: ", len(completed_tasks))

                    in_process_tasks.remove(fut.task_)
                    print(len(remaining_tasks)," TASKS REMAINING  --- ", len(in_process_tasks)," TASKS IN PROCESS  --- ",
                        len(completed_tasks), " COMPLETED TASKS")


            # Rethink this, do you really need to get rl everytime and why do you need n_excess_cores_free if you recalculate it later
            rl = flux.resource.list.resource_list(handle).get()
            n_cores_free = rl.free.ncores
            n_gpus_free = rl.free.ngpus
            # n_excess_cores_free = n_cores_free - n_gpus_free


            # print("PREPARING B.CSV FOR THE FIT")
            if (trigger_fit == 1) and (len(in_process_featurizations)==0):
                # Filesystem is slow consider that
                print("Preparing b.csv for the fit...")
                os.chdir("vasp-energy")
                new_completed_tasks = ["vasp-em_%i/b" % job_id for job_id in completed_tasks if job_id not in job_ids_for_fit]
                print(" ".join(new_completed_tasks))
                os.system("cat " + " ".join(new_completed_tasks) + " >> " + start_path + "features/b.csv")
                os.chdir("..")
                job_ids_for_fit = copy.copy(completed_tasks)
                trigger_fit = 2


            # print("SCHEDULING FITTING TASKS")
            if (trigger_fit == 2 and len(remaining_fits) > 0) and (len(in_process_featurizations)==0):
                #save to a file the configurations that have energies already from completed tasks
                # n_excess_cores_free = rl.free.ncores - 2*rl.free.ngpus - len(rs.nodelist)
                ncores_free = all_ncores - 2*all_ngpus - len(in_process_featurizations)*ncores_per_featurization
                ncores_free -= len(in_process_fits)*ncores_per_fit + len(in_process_costs) + len(rs.nodelist)
                while ncores_free>=ncores_per_fit and len(remaining_fits)>0:  # and (len(in_process_fits)<((all_ncores-all_ngpus)//ncores_per_fit)):
                    print("Starting the fits...")
                    i = remaining_fits.pop(0)
                    fit_directory = start_path + "fits/" + str(len(job_ids_for_fit))
                    if not os.path.isdir(fit_directory):
                        os.mkdir(fit_directory)
                    if mlip == "ACE":
                        fit_directory += "/" + ace_hyperparameters_to_string(hyperparameters_list[i], delimiter='_')
                    elif mlip == "SNAP":
                        fit_directory += "/" + snap_hyperparameters_to_string(hyperparameters_list[i], delimiter='_')
                    if not os.path.isdir(fit_directory):
                        os.mkdir(fit_directory)
                    fs = exe.submit(fit, start_path+"features/", hyperparameters_list[i], feature_names, mlip,
                                    resource_dict={"cores": 1, "threads_per_core": ncores_per_fit,
                                                   "gpus_per_core": 0, "num_nodes": 1, "cwd": fit_directory})
                    fs.task_ = i
                    fitting_futures.add(fs)
                    in_process_fits.append(i)
                    ncores_free -= ncores_per_fit
                
                if len(remaining_fits) == 0:
                    if len(remaining_tasks) != 0 or len(in_process_tasks) != 0 or wait_for_last_fit == 1:
                        trigger_fit = 0
                        remaining_fits = [i for i in range(len(hyperparameters_list))]
                    else:
                        trigger_fit = 0


            # print("SCHEDULING COST TASKS")
            if pareto_mode:
                ncores_free = all_ncores - 2*all_ngpus - len(in_process_featurizations)*ncores_per_featurization
                ncores_free -= len(in_process_fits)*ncores_per_fit + len(in_process_costs) + len(rs.nodelist)
                nconfigs4cost = config["MODE"]["nconfigurations_for_cost"]
                while ncores_free>=1 and len(remaining_costs)>0:
                    print("Starting the cost estimation...")
                    i = remaining_costs.pop(0)
                    costs_directory = start_path + "costs/"
                    if mlip == "ACE":
                        costs_directory += ace_hyperparameters_to_string(hyperparameters_list_noeweight[i], delimiter='_', w_eweight=False)
                    elif mlip == "SNAP":
                        costs_directory += snap_hyperparameters_to_string(hyperparameters_list_noeweight[i], delimiter='_', w_eweight=False)
                    if not os.path.isdir(costs_directory): os.mkdir(costs_directory)
                    fs = exe.submit(featurize, df["ase_atoms"].sample(n=nconfigs4cost,random_state=42).to_list(), config,
                                    fitsnap_config, rcuts, only_cost=True, resource_dict={"cores": 1, "gpus_per_core": 0,
                                                                                           "num_nodes": 1, "cwd": costs_directory})
                    fs.task_ = i
                    cost_futures.add(fs)
                    in_process_costs.append(i)
                    ncores_free -= 1


            # print("PROCESSING FITSNAP FUTURES")
            if feature_mode:
                featurizations_done, featurization_futures = concurrent.futures.wait(featurization_futures, timeout=0.1)
                for fut in featurizations_done:
                    feature_names = fut.result()[0]
                    print(len(feature_names),feature_names)
                    completed_featurizations.append(fut.task_)
                    in_process_featurizations.remove(fut.task_)
                    print(len(remaining_featurizations)," FEATURIZATIONS REMAINING  --- ", len(in_process_featurizations)," FEATURIZATIONS IN PROCESS  --- ",
                        len(completed_featurizations), " COMPLETED FEATURIZATIONS")


            # print("PROCESSING FITTING FUTURES")
            if fit_mode:
                fitting_done, fitting_futures = concurrent.futures.wait(fitting_futures, timeout=0.1)
                for fut in fitting_done:
                    completed_fits.append(fut.task_)
                    in_process_fits.remove(fut.task_)
                    print(len(remaining_fits)," FITS REMAINING  --- ", len(in_process_fits)," FITS IN PROCESS  --- ",
                        len(completed_fits), " COMPLETED FITS")


            # print("PROCESSING COSTS FUTURES")
            if pareto_mode:
                costs_done, cost_futures = concurrent.futures.wait(cost_futures, timeout=0.1)
                for fut in costs_done:
                    completed_costs.append(fut.task_)
                    in_process_costs.remove(fut.task_)
                    print(len(remaining_costs)," COSTS REMAINING  --- ", len(in_process_costs)," COSTS IN PROCESS  --- ",
                        len(completed_costs), " COMPLETED COSTS")


            # print("DOING PARETO FRONT")
            if len(completed_fits)==len(hyperparameters_list) and len(completed_costs)==len(hyperparameters_list_noeweight):
                completed_fits = []
                print("All fits are done!")
                if pareto_mode:
                    if pareto(tasks, start_path, hyperparameters_list, hyperparameters_list_noeweight, feature_names, mlip, 
                              job_ids_for_fit, remaining_fits, trigger_fit, auto_reduce_hps, wait_for_last_fit):
                        break
            
            
            # print("TRIGGERING LAST FIT")
            if wait_for_last_fit and len(in_process_fits) == 0:
                trigger_fit = 1
                wait_for_last_fit = 0
                print("Triggering last fit: ",len(completed_tasks))

            
            # if len(in_process_featurizations) == 0:
            with open("checkpoint.pkl", "wb") as f:
                pickle.dump((completed_featurizations, completed_tasks, completed_fits, completed_costs, job_ids_for_fit,
                             feature_names, trigger_fit, wait_for_last_fit), f)


if __name__ == "__main__":
    main()