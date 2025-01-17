# grace-data

This repo contains all experiments logs and the script we use to plot the data.

- `GRACE_logs_release.zip.00*`: All experiments logs.

- `GRACE_data_release.pickle`: Extracted experiments data in pandas dataframe format, each entry corresponds to one log in `GRACE_logs_release.zip.00*` with the same `id` tag, e.g. `g7c2ij800`.

- `GRACE_notebook_release.ipynb`: The python notebook to plot the data.

- `pull_from_wandb.py`: The script to download the logs (same as `GRACE_logs_release.zip.00*`) from our database and process them into pandas dataframe (same as `GRACE_data_release.pickle`). 

  With all these files, you will be able to generate exact same figures in our GRACE paper.
  

