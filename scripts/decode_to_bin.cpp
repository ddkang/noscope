#include <iostream>
#include <opencv2/opencv.hpp>

const size_t kResol = 50;
const size_t kFrameSize = kResol * kResol * 3;
const size_t kTrainFrames = 400000;

std::random_device rd;
std::mt19937 gen(rd());

int main(int argc, char* argv[]) {
  std::vector<uint8_t> data(kFrameSize * kTrainFrames);

  std::string fname(argv[1]);
  cv::VideoCapture cap(fname);
  cv::Mat frame, resized;
  for (size_t i = 0; i < kTrainFrames; i++) {
    cap >> frame;
    cv::resize(frame, resized, cv::Size(kResol, kResol), 0, 0, cv::INTER_NEAREST);
    memcpy(&data[i * kFrameSize], resized.data, kFrameSize);
  }

  std::ofstream fout(argv[2], std::ios::binary | std::ios::out);
  fout.write((char *) &data[0], kFrameSize * kTrainFrames);
  fout.close();
}
