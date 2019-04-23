#! /bin/bash

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  TARGET="$(readlink "$SOURCE")"
  if [[ $TARGET == /* ]]; then
    SOURCE="$TARGET"
  else
    DIR="$( dirname "$SOURCE" )"
    SOURCE="$DIR/$TARGET" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
  fi
done
DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )" # script dir
PARENT_DIR="$( dirname "$DIR" )" # parent of script dir

IFS=':' read -ra PATHS <<< "$ROS_PACKAGE_PATH" # split $ROS_PACKAGE_PATH string into an array

FOUND=""
for i in "${PATHS[@]}"; do
  if [ "$i" = "$PARENT_DIR" ]; then
    FOUND="$PARENT_DIR"
    break;
  fi
done

if [ -z "$FOUND" ]; then
  export "ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$PARENT_DIR" # ensure $ROS_PACKAGE_PATH contains $PARENT_DIR
fi

if [ ! -d "build" ]; then
  mkdir build
fi
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release "$@"
make -j8
