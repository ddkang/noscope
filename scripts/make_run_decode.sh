g++ -O3 -std=c++11 decode_to_bin.cpp -L/usr/local/lib -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc
BASE_NAME="taipei"
time ./a.out \
    "/root/infolab/fabuzaid/vuse-datasets-completed/videos/$BASE_NAME.mp4" \
    "/root/bailis/ddkang/noscope/data/resol-50-videos/$BASE_NAME.bin"
