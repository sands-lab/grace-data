#pull from wandb
import os

os.environ["WANDB_API_KEY"] = "INSERT YOUR WANDB API KEY"

import wandb
import os
import sys
import re
import pandas as pd
import numpy as np

PROJECTS = ["cifar10-classification", "imagenet-classification", "ml-20m-recommendation",
           "PTB-language_modeling", "DAGM2007-segmentation", "cifar10-classification-pytorch"]
METHODS = ['none', 'fp16', 'randomk', 'topk', 'threshold', 'terngrad', 'qsgd', 'dgc', 'adaq', 'signsgd', 'efsignsgd', 'signum', 'adas', 'onebit', 'powersgd', '8bit', 'natural', 'sketch', 'inceptionn']
GPUS = ['gtx1080ti', 'p100', 'v100']
DATASETS = ['cifar10', 'imagenet', 'ml-20m', 'DAGM2007', 'PTB']
ENVS = ['ibex', 'mcnodes']
COMM_METHODS = ['comm-method-allgather', 'comm-method-allreduce']

def row_filter(df,cond):
    conds = [df[k] == v for k,v in cond.items()]
    c = conds[0]
    for x in conds:
        c = c & x
    return df[c]

def download_logs(runs):
    ok_logs = []
    no_logs = []

    cwd = os.getcwd()
    logs = os.path.join(cwd, "logs")
    if not os.path.isdir(logs):
        os.mkdir(logs)
    os.chdir(logs)

    for run in runs:
        if 'test' in run.tags:
            continue

        rpath = os.path.join(logs, run.id)
        rfile = os.path.join(logs, run.id,"output.log")
        if os.path.isfile(rfile):
            continue
        try:
            file = run.file("output.log")
            if not os.path.isdir(rpath):
                os.mkdir(rpath)
            os.chdir(rpath)
            file.download()
            ok_logs.append(run.id)
        except wandb.apis.CommError as e:
            print("run %s has no output.log file" % run.id, file=sys.stderr)
            no_logs.append(run.id)
        finally:
            os.chdir(logs)
    os.chdir(cwd)

    return ok_logs, no_logs

def has_all_tags(tags, in_tags):
    for t in in_tags:
        if t not in tags:
            return False
    return True

def has_no_tags(tags, out_tags):
    for t in out_tags:
        if t in tags:
            return False
    return True

def filter(runs, out_ids = None, in_tags = None, out_tags=None):
    if out_ids is not None:
        runs = [r for r in runs if r.id not in out_ids]
    if in_tags is not None:
        runs = [r for r in runs if has_all_tags(r.tags, in_tags)]
    if out_tags is not None:
        runs = [r for r in runs if has_no_tags(r.tags, out_tags)]
    return runs

def run_to_log(id):
    cwd = os.getcwd()
    return os.path.join(cwd, "logs", id, "output.log")

def tabulate(runs, project):
    data = []
    for run in runs:
        # if run.state != 'finished':
        #     print("run %s state is %s; skipping not finished" % (run.id, run.state), file=sys.stderr)
        #     continue

        tags = run.tags

        if 'test' in tags:
            continue
        # meta data
        methods = [m for m in METHODS if m in tags]
        if len(methods) == 0:
            print("run %s has no method tag" % run.id, file=sys.stderr)
            continue
        elif len(methods) > 1:
            print("run %s has more than one method tag: %s" % (run.id, str(methods)), file=sys.stderr)
            continue
        method = methods[0]

        datasets = [m for m in DATASETS if m in tags]
        if len(datasets) == 0:
            print("run %s has no dataset tag" % run.id, file=sys.stderr)
            continue
        elif len(datasets) > 1:
            print("run %s has more than one dataset tag: %s" % (run.id, str(datasets)), file=sys.stderr)
            continue
        dataset = datasets[0]

        envs = [m for m in ENVS if m in tags]
        if len(envs) == 0:
            print("run %s has no env tag" % run.id, file=sys.stderr)
            continue
        elif len(envs) > 1:
            print("run %s has more than one env tag: %s" % (run.id, str(envs)), file=sys.stderr)
            continue
        env = envs[0]

        job_id = "n/a"
        if env == "ibex" and "SLURM_JOB_ID" in run.config:
            job_id = run.config.get("SLURM_JOB_ID")

        gpus = [m for m in GPUS if m in tags]
        if len(gpus) == 0:
            if "mcnodes" in tags:
                gpus = ["v100"]
            else:
                print("run %s has no gpu_type tag" % run.id, file=sys.stderr)
                continue
        elif len(gpus) > 1:
            print("run %s has more than one gpu_type tag: %s" % (run.id, str(gpus)), file=sys.stderr)
            continue
        gpu_type = gpus[0]

        if '8gpu' in tags:
            workers = 8
        elif '4gpu' in tags:
            workers = 4
        elif '2gpu' in tags:
            workers = 2
        elif '1gpu' in tags:
            workers = 1
        else:
            print("run %s has no *gpu tag" % run.id, file=sys.stderr)
            continue

        if 'use-memory-true' in tags:
            memory = 'true'
        elif 'use-memory-false' in tags:
            memory = 'false'
        else:
            if method != 'none' and method != 'fp16':
                print("run %s has no use-memory-* tag" % run.id, file=sys.stderr)
                continue
            memory = "n/a"

        if 'bw-10g' in tags:
            bw = '10g'
        elif 'bw-25g' in tags:
            bw = '25g'
        elif '1g' in tags:
            bw = '1g'
        elif '10g' in tags:
            bw = '10g'
        elif '25g' in tags:
            bw = '25g'
        else:
            if 'mcnodes' in tags:
                bw = '25g'
            else:
                bw = "n/a"
            bw = "n/a"

        ratios = [k for k in tags if k.startswith('compress-ratio-')]
        if len(ratios) > 1:
            print("run %s has more than one compress-ratio-* tag: %s" % (run.id, str(ratios)), file=sys.stderr)
            continue
        elif len(ratios) == 1:
            ratio = ratios[0][15:]
        else:
            ratio = "n/a"

        run_no = 0
        if 'run1' in tags:
            run_no = 1
        elif 'run2' in tags:
            run_no = 2
        elif 'run3' in tags:
            run_no = 3
        elif 'run4' in tags:
            run_no = 4
        elif 'run5' in tags:
            run_no = 5

        comm_method = "n/a"
        if 'comm-method-allreduce' in tags:
            comm_method = 'allreduce'
        elif 'comm-method-allgather' in tags:
            comm_method = 'allgather'

        mode = 'accuracy'
        if 'throughput' in tags:
            mode = 'throughput'
        elif 'data-volume' in tags:
            mode = 'data-volume'
        if 'upper-bound' in tags:
            mode = 'upper-bound'
        if 'wall_time' in tags:
            mode = 'wall_time'

        backend = 'mpi'
        if 'nccl' in tags:
            backend = 'nccl'

        name = run.name
        s = name.split('_')
        if len(s) > 1 and s[-1] in METHODS:
            name = s[-1] + '_' + '_'.join(s[0:-1])

        if project in ["cifar10-classification", "imagenet-classification"]:
            data_volume = "n/a"
            if 'data-volume' in tags:
                acc, has_nccl = parse_classification_debug_log(run_to_log(run.id))
                data_volume = np.mean(acc.series())
            acc, acc_top1, _ = parse_classification_throughput_log(run_to_log(run.id))
            throughput = np.mean(acc.series()[-100:-1]) * workers  #acc.series()[-1] is the total images/s in log

            
            max_top1 = 'n/a'
            if mode in ['accuracy', 'wall_time']:
                if len(acc_top1.series()) > 1:
                    max_top1 = np.max(acc_top1.series())
                else:
                    max_top1 = 0
                    print("run %s has no top1 accuracy found in log, max_top1_accuracy is set to 0" % run.id)
            
            r = [name,
                 mode,
                 backend,
                 method,
                 ratio,
                 memory,
                 workers,
                 dataset,
                 env,
                 gpu_type,
                 bw,
                 comm_method,
                 run.config.get("model"),
                 run.config.get("optimizer"),
                 run.config.get("momentum"),
                 run.id,
                 run_no,
                 job_id,
                 run.summary.get("global_step"),
                 run.summary.get("_runtime"),
                 run.summary.get("total_images_per_sec"),
                 #run.summary.get("eval/Accuracy@1"),
                 max_top1,
                 run.summary.get("eval/Accuracy@5"),
                 run.summary.get("total_loss"),
                 data_volume,
                 throughput,
                ]
            cols=["name", "mode","backend", "method", "ratio", "memory", "workers", "dataset", "env", "gpu_type", 
                  "bw", "comm_method", "model", "optimizer", "momentum", "id", "run_no", "job_id", "step", 
                  "runtime", "total_images_per_sec", "top_1_accuracy", "top_5_accuracy", "loss", "data_volume","throughput"]
        
        elif project == 'ml-20m-recommendation':
            model = "ncf"
            optimizer = "adam"
            data_volume = "n/a"
            if 'data-volume' in tags:
                first_to_target, best_hr, best_epoch, has_nccl, acc = parse_ncf_debug_log(run_to_log(run.id))
                data_volume = np.mean(acc.series())
            else:
                first_to_target, best_hr, best_epoch, has_nccl = parse_ncf_log(run_to_log(run.id))
                
            #if comm_method == 'allreduce' and not has_nccl:
            #   print("run %s has allreduce and does not use NCCL: %d %s" % (run.id, run_no, name), file=sys.stderr)
            #   continue
            r = [name,
                 mode,
                 backend,
                 method,
                 ratio,
                 memory,
                 workers,
                 dataset,
                 env,
                 gpu_type,
                 bw,
                 comm_method,
                 model,
                 optimizer,
                 run.id,
                 run_no,
                 job_id,
                 run.summary.get("epoch"),
                 run.summary.get("_runtime"),
                 run.summary.get("train/total_throughput"),
                 run.summary.get("eval/hit_rate"),
                 run.summary.get("eval/ndcg"),
                 first_to_target,
                 best_hr,
                 best_epoch,
                 data_volume,
                ]
            cols=["name", "mode","backend", "method", "ratio", "memory", "workers", "dataset", "env", "gpu_type", "bw", 
                  "comm_method", "model", "optimizer", "id", "run_no", "job_id", "epoch", "runtime", 
                  "total_throughput", "hit_rate", "ndcg", "first_to_target", "best_hit_rate", "best_epoch", "data_volume"]

        elif project == 'PTB-language_modeling':
            model = "LSTM"
            optimizer = "SGD"
            throughput = parse_pbt_log(run_to_log(run.id)) * workers
            data_volume = "n/a"
            if 'data-volume' in tags:
                acc = parse_ptb_debug_log(run_to_log(run.id))
                data_volume = np.mean(acc.series())
                
            r = [name,
                 mode,
                 backend,
                 method,
                 ratio,
                 memory,
                 workers,
                 dataset,
                 env,
                 gpu_type,
                 bw,
                 comm_method,
                 model,
                 optimizer,
                 run.id,
                 run_no,
                 job_id,
                 run.summary.get("_step"),
                 run.summary.get("global_step"),
                 run.summary.get("_runtime"),
                 run.summary.get("cost"),
                 run.summary.get("test_cost"),
                 run.summary.get("validation_cost"),
                 run.summary.get("perplexity"),
                 run.summary.get("test_perplexity"),
                 run.summary.get("validation_perplexity"),
                 throughput,
                 data_volume,
                ]
            cols=["name", "mode","backend", "method", "ratio", "memory", "workers", "dataset", 
                  "env", "gpu_type", "bw", "comm_method", "model", "optimizer", "id", "run_no", "job_id", 
                  "step", "global_step", "runtime", "cost", "test_cost", "validation_cost", 
                  "perplexity", "test_perplexity", "validation_perplexity", "throughput", "data_volume"]

        elif project == 'DAGM2007-segmentation':
            model = "unet"
            optimizer = "RMSProp"
            res = parse_segmentation_log(run_to_log(run.id))
            data_volume = "n/a"
            if 'data-volume' in tags:
                acc = parse_segmentation_debug_log(run_to_log(run.id))
                data_volume = np.mean(acc.series())
            r = [name,
                 mode,
                 backend,
                 method,
                 ratio,
                 memory,
                 workers,
                 dataset,
                 env,
                 gpu_type,
                 bw,
                 comm_method,
                 model,
                 optimizer,
                 run.id,
                 run_no,
                 job_id,
                 run.summary.get('_runtime'),
                 run.summary.get('_step'),
                 run.summary.get('global_step'),
                 run.summary.get('train/total_throughput'),
#                  run.summary.get('Confusion_Matrix/false_positives_1'),
#                  run.summary.get('Confusion_Matrix/true_positives_1'),
#                  run.summary.get('Confusion_Matrix/true_negatives_1'),
#                  run.summary.get('Confusion_Matrix/false_negatives_1'),
                 res['IoU_THS_0.05:'],
                 res['IoU_THS_0.125:'],
                 res['IoU_THS_0.25:'],
                 res['IoU_THS_0.5:'],
                 res['IoU_THS_0.75:'],
                 res['IoU_THS_0.85:'],
                 res['IoU_THS_0.95:'],
                 res['IoU_THS_0.99:'],
                 data_volume,
                ]
            cols=["name", "mode","backend", "method", "ratio", "memory", "workers", "dataset", 
                  "env", "gpu_type", "bw", "comm_method", "model", "optimizer", "id", "run_no", "job_id", 
                "runtime",
                "step",
                "global_step",
                "throughput",
#                 "Confusion_Matrix/false_positives_1",
#                 "Confusion_Matrix/true_positives_1",
#                 "Confusion_Matrix/true_negatives_1",
#                 "Confusion_Matrix/false_negatives_1",
                "IoU_Metrics/iou_score_ths_0.05",
                "IoU_Metrics/iou_score_ths_0.125",
                "IoU_Metrics/iou_score_ths_0.25",
                "IoU_Metrics/iou_score_ths_0.5",
                "IoU_Metrics/iou_score_ths_0.75",
                "IoU_Metrics/iou_score_ths_0.85",
                "IoU_Metrics/iou_score_ths_0.95",
                "IoU_Metrics/iou_score_ths_0.99",
                  "data_volume",
                 ]
            
        elif project == 'cifar10-classification-pytorch':
            model = "Resnet9"
            optimizer = "momentum"

            RDMA = 'false'
            if 'rdma' in tags:
                RDMA = 'true'
            r = [name,
                 mode,
                 backend,
                 method,
                 ratio,
                 memory,
                 workers,
                 dataset,
                 env,
                 gpu_type,
                 bw,
                 comm_method,
                 model,
                 optimizer,
                 run.id,
                 run_no,
                 job_id,
                run.summary.get('epoch'),
                run.summary.get('test_acc'),
                run.summary.get('_runtime'),
                run.summary.get('total_train_throughput'),
                run.summary.get('train_loss'),
                RDMA,
                ]
            cols=["name", "mode","backend", "method", "ratio", "memory", "workers", "dataset", 
                  "env", "gpu_type", "bw", "comm_method", "model", "optimizer", "id", "run_no", "job_id", 
                  "epoch", "top_1_accuracy", "runtime", "throughput", "loss","RDMA"]
            
        data.append(r)
        
    return pd.DataFrame(data, columns=cols)

def data_volumes(runs, project):
    data = []
    for run in runs:
        if run.state != 'finished':
            print("run %s state is %s; skipping not finished" % (run.id, run.state), file=sys.stderr)
            continue

        tags = run.tags

        if 'test' in tags:
            continue

        name = run.name
        s = name.split('_')
        if len(s) > 1 and s[-1] in METHODS:
            name = s[-1] + '_' + '_'.join(s[0:-1])

        if "data-volume" not in tags:
            print("run %s has no data-volume tag" % run.id, file=sys.stderr)
            continue
            
        if project == 'ml-20m-recommendation':
            first_to_target, best_hr, best_epoch, has_nccl, acc = parse_ncf_debug_log(run_to_log(run.id))
            r = [name, np.mean(acc.series())]
            cols = ["name", "data_volume"]
            
        elif project in ["cifar10-classification", "imagenet-classification"]:
            acc, has_nccl = parse_classification_debug_log(run_to_log(run.id))
            r = [name, run.id, run.config.get("model"), np.mean(acc.series())]
            cols = ["name", "id", "model", "data_volume"]
            
        elif project == 'PTB-language_modeling':
            acc = parse_ptb_debug_log(run_to_log(run.id))
            r = [name, run.id, run.config.get("model"), np.mean(acc.series())]
            cols = ["name", "id", "model", "data_volume"]
            
        elif project == 'DAGM2007-segmentation':
            acc = parse_segmentation_debug_log(run_to_log(run.id))
            r = [name, run.id, run.config.get("model"), np.mean(acc.series())]
            cols = ["name", "id", "model", "data_volume"]
        else:
            print("run %s not parsed for data volume" % run.id, file=sys.stderr)
            continue

        data.append(r)

    return pd.DataFrame(data, columns=cols)

def throughputs(runs, project):
    data = []
    for run in runs:
        if run.state != 'finished':
            print("run %s state is %s; skipping not finished" % (run.id, run.state), file=sys.stderr)
            continue

        tags = run.tags

        if 'test' in tags:
            continue

        name = run.name
        s = name.split('_')
        if len(s) > 1 and s[-1] in METHODS:
            name = s[-1] + '_' + '_'.join(s[0:-1])

        methods = [m for m in METHODS if m in tags]
        if len(methods) == 0:
            print("run %s has no method tag" % run.id, file=sys.stderr)
            continue
        elif len(methods) > 1:
            print("run %s has more than one method tag: %s" % (run.id, str(methods)), file=sys.stderr)
            continue
        method = methods[0]

        if "throughput" not in tags and "cifar10" not in tags:
            print("run %s has no throughput tag" % run.id, file=sys.stderr)
            continue

        if 'upper-bound' in tags:
            mode = 'upper-bound'
        else:
            mode = 'throughput'

        if 'bw-10g' in tags:
            bw = '10g'
        elif 'bw-25g' in tags:
            bw = '25g'
        elif '1g' in tags:
            bw = '1g'
        elif '10g' in tags:
            bw = '10g'
        elif '25g' in tags:
            bw = '25g'
        else:
            if 'mcnodes' in tags:
                bw = '25g'
            else:
                bw = "n/a"
            bw = "n/a"

        if '8gpu' in tags:
            workers = 8
        elif '4gpu' in tags:
            workers = 4
        elif '2gpu' in tags:
            workers = 2
        elif '1gpu' in tags:
            workers = 1
        else:
            print("run %s has no *gpu tag" % run.id, file=sys.stderr)
            continue

        if "classification" in tags:
            acc, acc_top1, has_nccl = parse_classification_throughput_log(run_to_log(run.id))
            series = acc.series()
            # skip first 100 iterations
            series = series[100:len(series)]
            # r = [name, run.id, mode, run.config.get("model"), bw, workers, run.summary.get("train/total_throughput")]
            r = [name, run.id, method, mode, run.config.get("model"), bw, workers, run.summary.get("total_images_per_sec"), run.summary.get("_runtime")]
            cols = ["name", "id", "method", "mode", "model", "bw", "workers", "throughput", "runtime"]
        else:
            print("run %s not parsed for throughput" % run.id, file=sys.stderr)
            continue

        data.append(r)

    return pd.DataFrame(data, columns=cols)

 
def parse_classification_throughput_log(fname):
    with open(fname, "r") as f:
        lines = [line for line in f]

    class AccThroughput:
        def __init__(self):
            self.a = []

        def acc(self, val):
            self.a.append(val)

        def series(self):
            return np.array(self.a)

    acc = AccThroughput()
    acc_top1 = AccThroughput()
    has_nccl = False
    for line in lines:
        if 'NCCL' in line:
            has_nccl = True
        if 'images/sec:' in line:
            components = line.split()
            acc.acc(float(components[2]))
        elif 'Accuracy @ 1' in line:
            components = line.split()
            acc_top1.acc(float(components[4]))

    return acc, acc_top1, has_nccl

def parse_classification_debug_log(fname):
    with open(fname, "r") as f:
        lines = [line for line in f]
    has_nccl = False
    tensors_no = 0
    acc = None

    class AccVolume:
        def __init__(self, tensors_no):
            self.tensors_no = tensors_no
            self.a = []
            self.cur_a = 0
            self.cur_cnt = 0

        def acc(self, size):
            self.cur_a += size
            self.cur_cnt += 1
            if self.cur_cnt == self.tensors_no:
                self.a.append(self.cur_a)
                self.cur_a = 0
                self.cur_cnt = 0

        def series(self):
            return np.array(self.a)

    bundle = 1
    for line in lines:
        if '==Debug== tensor 0/' in line:
            x = re.search("tensor 0/([0-9]+)", line)
            if x is None:
                print("in %s txt %s did not match regex" % (fname, txt), file=sys.stderr)
                return
            bundle = int(x.group(1))
            break

    for line in lines:
        if '==Debug== The model has' in line:
            tensors_no = int(line.split()[4]) * bundle
            if acc is not None:
                print("in %s acc already not None" % fname, file=sys.stderr)
                return None
            acc = AccVolume(tensors_no)
            break

    if acc is None:
        tensors_tot = 0
        for line in lines:
            if 'Done warm up' in line:
                break
            if '==Debug== tensor' in line:
                txts = line.split('==Debug== ')
                txts.pop(0)
                for txt in txts:
                    x = re.search("<dtype: '([a-z0-9]+)'> size:\\[([0-9]+)\\]", txt)
                    if x is None:
                        print("in %s txt %s did not match regex" % (fname, txt), file=sys.stderr)
                        continue
                    tensors_tot += 1
        if tensors_tot % 10 != 0:
            print("in %s tensors_tot %d is not a multiple of 10" % (fname, tensors_tot), file=sys.stderr)
            return None

        # print("in %s tensors_tot %d (%d tensors per iteration)" % (fname, tensors_tot, tensors_tot / 10 / bundle), file=sys.stderr)

        if tensors_tot / 10 / bundle != 161:
            print("in %s tensors_tot %d (%d tensors per iteration) is not 161" % (fname, tensors_tot, tensors_tot / 10 / bundle), file=sys.stderr)

        acc = AccVolume(tensors_tot / 10) # tensor_no is tensor_tot / 10 warm up iterations

    if acc is None:
        print("in %s acc is None" % fname, file=sys.stderr)
        return None

    tensors_tot = 0
    done_warm_up = False
    zero_size_warn = False
    for line in lines:
        if 'NCCL' in line:
            has_nccl = True
        if 'Done warm up' in line:
            done_warm_up = True
        if not done_warm_up and '==Debug== tensor' in line:
            txts = line.split('==Debug== ')
            txts.pop(0)
            for txt in txts:
                x = re.search("<dtype: '([a-z0-9]+)'> size:\\[([0-9]+)\\]", txt)
                if x is None:
                    print("in %s txt %s did not match regex" % (fname, txt), file=sys.stderr)
                    continue
                t_type = x.group(1)
                t_size = int(x.group(2))
                if t_type == 'int8' or t_type == 'uint8' or t_type == 'bool':
                    pass
                elif t_type == 'int16' or t_type == 'uint16' or t_type == 'float16':
                    t_size *= 2
                elif t_type == 'int32' or t_type == 'float32':
                    t_size *= 4
                else:
                    print("in %s t_type %s is not known" % (fname, t_type), file=sys.stderr)
                    continue
                if t_size == 0 and not zero_size_warn:
                    zero_size_warn = True
                    print("in %s t_size is zero: %s" % (fname, line), file=sys.stderr)

                acc.acc(t_size)
                tensors_tot += 1
#    if tensors_tot % acc.tensors_no != 0:
#        print("in %s tensors_tot %d is not a multiple of %d" % (fname, tensors_tot, acc.tensors_no), file=sys.stderr)
#        return None
            

    return acc, has_nccl


def parse_ncf_log(fname):
    with open(fname, "r") as f:
        lines = [line for line in f]
    first_to_target = None
    best_hr = float('nan')
    best_epoch = float('nan')
    has_nccl = False
    for line in lines:
        if 'NCCL' in line:
            has_nccl = True
        if 'First Epoch to hit:' in line:
            x = line.split(':')[-1]
            if 'None' in x:
                continue
            first_to_target = int(x)
        elif 'Best HR:' in line:
            best_hr = float(line.split(':')[-1])
        elif 'Best Epoch:' in line:
            best_epoch = float(line.split(':')[-1])
    return first_to_target, best_hr, best_epoch, has_nccl

def parse_ncf_debug_log(fname):
    with open(fname, "r") as f:
        lines = [line for line in f]
    first_to_target = None
    best_hr = float('nan')
    best_epoch = float('nan')
    has_nccl = False
    tensors_no = 0
    acc = None

    class AccVolume:
        def __init__(self, tensors_no):
            self.tensors_no = tensors_no
            self.a = []
            self.cur_a = 0
            self.cur_cnt = 0

        def acc(self, size):
            self.cur_a += size
            self.cur_cnt += 1
            if self.cur_cnt == self.tensors_no:
                self.a.append(self.cur_a)
                self.cur_a = 0
                self.cur_cnt = 0

        def series(self):
            return np.array(self.a)

    bundle = 1
    for line in lines:
        if '==Debug== tensor 0/' in line:
            x = re.search("tensor 0/([0-9]+)", line)
            if x is None:
                print("in %s txt %s did not match regex" % (fname, txt), file=sys.stderr)
                return None
            bundle = int(x.group(1))
            break

    for line in lines:
        if '==Debug== The model has' in line:
            tensors_no = int(line.split()[4]) * bundle
            if acc is not None:
                print("in %s acc already not None" % fname, file=sys.stderr)
                return None
            acc = AccVolume(tensors_no)
            break

    if acc is None:
        print("in %s acc is None" % fname, file=sys.stderr)
        return None

    for line in lines:
        if 'NCCL' in line:
            has_nccl = True
        if 'First Epoch to hit:' in line:
            x = line.split(':')[-1]
            if 'None' in x:
                continue
            first_to_target = int(x)
        elif 'Best HR:' in line:
            best_hr = float(line.split(':')[-1])
        elif 'Best Epoch:' in line:
            best_epoch = float(line.split(':')[-1])
        if '==Debug== tensor' in line:
            txts = line.split('==Debug== ')
            txts.pop(0)
            for txt in txts:
                x = re.search("<dtype: '([a-z0-9]+)'> size:\\[([0-9]+)\\]", txt)
                if x is None:
                    print("in %s txt %s did not match regex" % (fname, txt), file=sys.stderr)
                    continue
                t_type = x.group(1)
                t_size = int(x.group(2))
                if t_type == 'int8' or t_type == 'uint8' or t_type == 'bool':
                    pass
                elif t_type == 'int16' or t_type == 'uint16' or t_type == 'float16':
                    t_size *= 2
                elif t_type == 'int32' or t_type == 'float32':
                    t_size *= 4
                else:
                    print("in %s t_type %s is not known" % (fname, t_type), file=sys.stderr)
                    continue
                acc.acc(t_size)
            

    return first_to_target, best_hr, best_epoch, has_nccl, acc


def parse_pbt_log(fname):
    with open(fname, "r") as f:
        lines = [line for line in f]
    throughput = 0
    for line in lines:
        if '|##############################' in line:
            x = line.split(',')[1]
            try:
                temp = float(x.split('it/s')[0])
            except:
                print("fname: %s has error: %s %s" % (fname,x,x.split('it/s')[0]))
            throughput = temp if temp > throughput else throughput
        if 'finished' in line:
            return throughput
    
    print("fname: %s has no throughput" % fname)
    return 'n/a'

def parse_ptb_debug_log(fname):
    with open(fname, "r") as f:
        lines = [line for line in f]

    tensors_no = 6
    acc = None

    class AccVolume:
        def __init__(self, tensors_no):
            self.tensors_no = tensors_no
            self.a = []
            self.cur_a = 0
            self.cur_cnt = 0

        def acc(self, size):
            self.cur_a += size
            self.cur_cnt += 1
            if self.cur_cnt == self.tensors_no:
                self.a.append(self.cur_a)
                self.cur_a = 0
                self.cur_cnt = 0

        def series(self):
            return np.array(self.a)

    bundle = 1
    for line in lines:
        if '==Debug== tensor 0/' in line:
            x = re.search("tensor 0/([0-9]+)", line)
            if x is None:
                print("in %s txt %s did not match regex" % (fname, txt), file=sys.stderr)
                #return None
            bundle = int(x.group(1))
            break

    acc = AccVolume(tensors_no)

    for line in lines:
        if '==Debug== tensor' in line:
            txts = line.split('==Debug== ')
            txts.pop(0)
            for txt in txts:
                x = re.search("<dtype: '([a-z0-9]+)'> size:\\[([0-9]+)\\]", txt)
                if x is None:
                    print("in %s txt %s did not match regex" % (fname, txt), file=sys.stderr)
                    continue
                t_type = x.group(1)
                t_size = int(x.group(2))
                if t_type == 'int8' or t_type == 'uint8' or t_type == 'bool':
                    pass
                elif t_type == 'int16' or t_type == 'uint16' or t_type == 'float16':
                    t_size *= 2
                elif t_type == 'int32' or t_type == 'float32':
                    t_size *= 4
                else:
                    print("in %s t_type %s is not known" % (fname, t_type), file=sys.stderr)
                    continue
                acc.acc(t_size)
    return acc


def parse_segmentation_log(fname):
    with open(fname, "r") as f:
        lines = [line for line in f]
    find_eval = False
    res = {}
    for line in lines:
        if 'Evaluation Results:' in line:
            find_eval = True
        if find_eval and 'IoU_THS' in line:
            key = line.split(' ')[-2]
            val = line.split(' ')[-1]
            res[key] = float(val)                
    if len(res) < 8:
        print("fname: %s has no evaluation results" % fname)
    return res

def parse_segmentation_debug_log(fname):
    with open(fname, "r") as f:
        lines = [line for line in f]

    tensors_no = 0
    acc = None

    class AccVolume:
        def __init__(self, tensors_no):
            self.tensors_no = tensors_no
            self.a = []
            self.cur_a = 0
            self.cur_cnt = 0

        def acc(self, size):
            self.cur_a += size
            self.cur_cnt += 1
            if self.cur_cnt == self.tensors_no:
                self.a.append(self.cur_a)
                self.cur_a = 0
                self.cur_cnt = 0

        def series(self):
            return np.array(self.a)

    bundle = 1
    for line in lines:
        if '==Debug== tensor 0/' in line:
            x = re.search("tensor 0/([0-9]+)", line)
            if x is None:
                print("in %s txt %s did not match regex" % (fname, txt), file=sys.stderr)
                #return None
            bundle = int(x.group(1))
            break

    for line in lines:
        if '==Debug== The model has' in line:
            tensors_no = int(line.split()[4]) * bundle
            if acc is not None:
                print("in %s acc already not None" % fname, file=sys.stderr)
                #return None
            tensors_no = int(line.split()[4]) * bundle # 69 for powersgd
    acc = AccVolume(tensors_no)

    for line in lines:
        if '==Debug== tensor' in line:
            txts = line.split('==Debug== ')
            txts.pop(0)
            for txt in txts:
                x = re.search("<dtype: '([a-z0-9]+)'> size:\\[([0-9]+)\\]", txt)
                if x is None:
                    print("in %s txt %s did not match regex" % (fname, txt), file=sys.stderr)
                    continue
                t_type = x.group(1)
                t_size = int(x.group(2))
                if t_type == 'int8' or t_type == 'uint8' or t_type == 'bool':
                    pass
                elif t_type == 'int16' or t_type == 'uint16' or t_type == 'float16':
                    t_size *= 2
                elif t_type == 'int32' or t_type == 'float32':
                    t_size *= 4
                else:
                    print("in %s t_type %s is not known" % (fname, t_type), file=sys.stderr)
                    continue
                acc.acc(t_size)
    return acc


def parse_log(fname):
    with open(fname, "r") as f:
        lines = [line for line in f]
    accu_1=[]
    accu_5=[]
    accu_steps=[]
    accu_time=[]
    loss=[]
    loss_steps=[]
    loss_time=[]
    for line in lines:
        if 'at global_step' in line:
            accu_steps.append(int(line.split()[-1]))
        elif "Accuracy" in line:
            components = line.split()
            accu_1.append(float(components[4]))
            accu_5.append(float(components[9]))
            accu_time.append(float(components[-1]))
        elif "images/sec:" in line:
            components = line.split()
            loss.append(float(components[8]))
            loss_time.append(float(components[9]))
            loss_steps.append(int(components[0]))
    return accu_1, accu_5, accu_time, accu_steps, loss, loss_steps, loss_time

def load(project, in_tags=None, out_tags=None):
    api = wandb.Api()
    runs = api.runs(f"sands-lab/{project}")

    ok_logs, no_logs = download_logs(runs)
    runs = filter(runs, out_ids=no_logs, in_tags=in_tags, out_tags=out_tags)

    return tabulate(runs, project)

def load_data_volumes(project, cfilter = None, in_tags=None, out_tags=None):
    api = wandb.Api()
    runs = api.runs(f"sands-lab/{project}")

    runs = filter(runs, in_tags=["data-volume"], out_tags=out_tags)

    if cfilter is not None:
        runs = [r for r in runs if cfilter(r)]

    if len(runs) == 0:
        print("load_data_volumes: all runs filtered out", file=sys.stderr)
        return None

    return data_volumes(runs, project)

def load_throughputs(project, cfilter = None, in_tags=None, out_tags=None):
    api = wandb.Api()
    runs = api.runs(f"sands-lab/{project}")

    # if project != "cifar10-classification":
    #     runs = filter(runs, in_tags=["throughput"])
    runs = filter(runs, in_tags=["throughput"])
    runs = filter(runs, in_tags=in_tags, out_tags=out_tags)

    if cfilter is not None:
        runs = [r for r in runs if cfilter(r)]

    if len(runs) == 0:
        print("load_throughputs: all runs filtered out", file=sys.stderr)
        return None

    return throughputs(runs, project)
