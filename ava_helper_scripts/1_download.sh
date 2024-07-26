DATA_DIR="./videos"
PARALLEL_DOWNLOAD_JOBS=10

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${DATA_DIR}
fi

wget https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt

cat ava_file_names_trainval_v2.1.txt | parallel -j ${PARALLEL_DOWNLOAD_JOBS} wget https://s3.amazonaws.com/ava-dataset/trainval/{} -P ${DATA_DIR}




