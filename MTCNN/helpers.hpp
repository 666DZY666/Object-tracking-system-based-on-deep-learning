#ifndef _HELPERS_HPP_
#define _HELPERS_HPP_

#include <chrono>
#include <opencv2/opencv.hpp>

class Timer {
private:
	std::chrono::time_point<std::chrono::system_clock> t1;
public:
	inline void start() {
		t1 = std::chrono::system_clock::now();
	} 
	inline double stop() {
		auto t2 = std::chrono::system_clock::now();
		return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	}
};

inline cv::Mat cropImage(cv::Mat img, cv::Rect r) {
	cv::Mat m = cv::Mat::zeros(r.height, r.width, img.type());
	int dx = std::abs(std::min(0, r.x));
	if (dx > 0) { r.x = 0; }
	r.width -= dx;
	int dy = std::abs(std::min(0, r.y));
	if (dy > 0) { r.y = 0; }	
	r.height -= dy;
	int dw = std::abs(std::min(0, img.cols - 1 - (r.x + r.width)));
	r.width -= dw;
	int dh = std::abs(std::min(0, img.rows - 1 - (r.y + r.height)));
	r.height -= dh;
	if (r.width > 0 && r.height > 0) {
		img(r).copyTo(m(cv::Range(dy, dy + r.height), cv::Range(dx, dx + r.width)));
	}
	return m;
}

inline void drawAndShowFace(cv::Mat img, cv::Rect r, const std::vector<cv::Point>& pts) {
	cv::Mat outImg;
	img.convertTo(outImg, CV_8UC3);
	cv::rectangle(outImg, r, cv::Scalar(0, 0, 255));
	for (size_t i = 0; i < pts.size(); ++i) {
		cv::circle(outImg, pts[i], 3, cv::Scalar(0, 0, 255));
	}
	cv::imshow("test", outImg);
	cv::waitKey(0);
}

#endif //_HELPERS_HPP_