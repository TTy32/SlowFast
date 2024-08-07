IN_DATA_DIR="./videos"
OUT_DATA_DIR="./videos_15min"
PARALLEL_JOBS=8


export OUT_DATA_DIR  # Export the variable

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

ls -A1 -U ${IN_DATA_DIR}/* | parallel -j ${PARALLEL_JOBS} '
  video=$(pwd)/{}
  echo -----------
  echo $video
  out_name="${OUT_DATA_DIR}/$(basename ${video})"
  echo $out_name
  if [ ! -f "${out_name}" ]; then
    ffmpeg -hwaccel cuda -ss 900 -t 901 -i "${video}" "${out_name}"
  fi
'
