IN_DATA_DIR="./videos_15min"
OUT_DATA_DIR="./frames"

PARALLEL_JOBS=10

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

export OUT_DATA_DIR  # Export variable to make it available for parallel jobs

process_video() {
  local video="$1"
  local video_name="${video##*/}"

  if [[ $video_name = *".webm" ]]; then
    video_name="${video_name::-5}"
  else
    video_name="${video_name::-4}"
  fi

  local out_video_dir="${OUT_DATA_DIR}/${video_name}/"
  mkdir -p "${out_video_dir}"

  local out_name="${out_video_dir}/${video_name}_%06d.jpg"

  ffmpeg -hwaccel cuda  -i "${video}" -r 30 -q:v 1 "${out_name}"
}

export -f process_video  # Export function to make it available for parallel jobs

find "${IN_DATA_DIR}" -type f | parallel -j ${PARALLEL_JOBS} process_video {}



