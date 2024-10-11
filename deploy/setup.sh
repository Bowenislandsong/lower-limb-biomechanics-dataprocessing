# create folders
mkdir -p /dataset/pandas /dataset/raw /local_results

# create links
ln -s /media/champagne/lower_limb_dataset/AB*.pkl ./dataset/pandas/
ln -s /media/champagne/lower_limb_dataset/AB*/ ./dataset/raw/   