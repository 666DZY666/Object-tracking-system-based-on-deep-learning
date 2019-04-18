#include "face_detector.hpp"
#include "helpers.hpp"

namespace mtcnn {

const std::string P_NET_PROTO = "/det1.prototxt";
const std::string P_NET_WEIGHTS = "/det1.caffemodel";
const std::string P_NET_REGRESSION_BLOB_NAME = "conv4-2";
const std::string P_NET_SCORE_BLOB_NAME = "prob1";
const float P_NET_WINDOW_SIDE = 12.f;
const int P_NET_STRIDE = 2;

const std::string R_NET_PROTO = "/det2.prototxt";
const std::string R_NET_WEIGHTS = "/det2.caffemodel";
const std::string R_NET_REGRESSION_BLOB_NAME = "conv5-2";
const std::string R_NET_SCORE_BLOB_NAME = "prob1";

const std::string O_NET_PROTO = "/det3.prototxt";
const std::string O_NET_WEIGHTS = "/det3.caffemodel";
const std::string O_NET_REGRESSION_BLOB_NAME = "conv6-2";
const std::string O_NET_SCORE_BLOB_NAME = "prob1";
const std::string O_NET_PTS_BLOB_NAME = "conv6-3";

const std::string L_NET_PROTO = "/det4.prototxt";
const std::string L_NET_WEIGHTS = "/det4.caffemodel";
const float L_THRESHOLD = 0.35f;

const float IMG_MEAN = 127.5f;
const float IMG_INV_STDDEV = 1.f / 128.f;

FaceDetector::FaceDetector(const std::string& modelDir, 
							float pThreshold, 
							float rThreshold, 
							float oThreshold, 
							bool useLNet,
							bool useGPU, 
							int deviceID) : pThreshold_(pThreshold), 
											rThreshold_(rThreshold), 
											oThreshold_(oThreshold),
											useLNet_(useLNet) {
	if (useGPU) {
		caffe::Caffe::set_mode(caffe::Caffe::GPU);
		caffe::Caffe::SetDevice(deviceID);
                printf("!!!GPU!!!\r\n");
	} else {
		caffe::Caffe::set_mode(caffe::Caffe::CPU);
                printf("CPU\r\n");
	}
	pNet_.reset( new caffe::Net<float> (modelDir + P_NET_PROTO, caffe::TEST));
	pNet_->CopyTrainedLayersFrom(modelDir + P_NET_WEIGHTS);
	rNet_.reset( new caffe::Net<float> (modelDir + R_NET_PROTO, caffe::TEST));
	rNet_->CopyTrainedLayersFrom(modelDir + R_NET_WEIGHTS);
	oNet_.reset( new caffe::Net<float> (modelDir + O_NET_PROTO, caffe::TEST));
	oNet_->CopyTrainedLayersFrom(modelDir + O_NET_WEIGHTS);
	if (useLNet_) {
		lNet_.reset( new caffe::Net<float> (modelDir + L_NET_PROTO, caffe::TEST));
		lNet_->CopyTrainedLayersFrom(modelDir + L_NET_WEIGHTS);
	}
}

std::vector<Face> FaceDetector::detect(cv::Mat img, float minFaceSize, float scaleFactor) {
	cv::Mat rgbImg;
	if (img.channels() == 3) {
		cv::cvtColor(img, rgbImg, CV_BGR2RGB);
	} else if (img.channels() == 4) {
		cv::cvtColor(img, rgbImg, CV_BGRA2RGB);
	}
	if (rgbImg.empty()) {
		return std::vector<Face>();
	}
	rgbImg.convertTo(rgbImg, CV_32FC3);
	rgbImg = rgbImg.t();
	std::vector<Face> faces = step1(rgbImg, minFaceSize, scaleFactor);
	faces = step2(rgbImg, faces);
	faces = step3(rgbImg, faces);
	if (useLNet_) {
		faces = step4(rgbImg, faces);
	}
	for (size_t i = 0; i < faces.size(); ++i) {
		std::swap(faces[i].bbox.x1, faces[i].bbox.y1);
		std::swap(faces[i].bbox.x2, faces[i].bbox.y2);
		for (int p = 0; p < NUM_PTS; ++p) {
			std::swap(faces[i].ptsCoords[2 * p], faces[i].ptsCoords[2 * p + 1]);
		}
	}
	return faces;
}

void FaceDetector::initNetInput(boost::shared_ptr< caffe::Net<float> > net, cv::Mat img) {
	std::vector<cv::Mat> channels;
	cv::split(img, channels);	
	caffe::Blob<float>* inputLayer = net->input_blobs()[0];
	assert(inputLayer->channels() == static_cast<int>(channels.size()));
	if (img.rows != inputLayer->height() || img.cols != inputLayer->width()) {
		inputLayer->Reshape(1, channels.size(), img.rows, img.cols);
		net->Reshape();
	}
	float* inputData = inputLayer->mutable_cpu_data();
	for (size_t i = 0; i < channels.size(); ++i) {
		channels[i] -= IMG_MEAN;
		channels[i] *= IMG_INV_STDDEV;
		memcpy(inputData, channels[i].data, sizeof(float) * img.cols * img.rows);
		inputData += img.cols * img.rows;
	}
}

void FaceDetector::initNetInput(boost::shared_ptr< caffe::Net<float> > net, std::vector<cv::Mat>& imgs) {
	caffe::Blob<float>* inputLayer = net->input_blobs()[0];
	float* inputData = inputLayer->mutable_cpu_data();
	for (size_t i = 0; i < imgs.size(); ++i) {
		std::vector<cv::Mat> channels;
		cv::split(imgs[i], channels);	
		for (size_t c = 0; c < channels.size(); ++c) {
			channels[c] -= IMG_MEAN;
			channels[c] *= IMG_INV_STDDEV;
			memcpy(inputData, channels[c].data, sizeof(float) * imgs[i].cols * imgs[i].rows);
			inputData += imgs[i].cols * imgs[i].rows;
		}
	}
}

std::vector<Face> FaceDetector::step1(cv::Mat img, float minFaceSize, float scaleFactor) {
	std::vector<Face> finalFaces;
	float maxFaceSize = static_cast<float>(std::min(img.rows, img.cols));
	float faceSize = minFaceSize;
	while (faceSize <= maxFaceSize) {
		float currentScale = (P_NET_WINDOW_SIDE) / faceSize;
		int imgHeight = std::ceil(img.rows * currentScale);
		int imgWidth = std::ceil(img.cols * currentScale);
		cv::Mat resizedImg;
		cv::resize(img, resizedImg, cv::Size(imgWidth, imgHeight), 0, 0, cv::INTER_AREA);
		initNetInput(pNet_, resizedImg);

		pNet_->Forward();
		const caffe::Blob<float>* regressionsBlob = pNet_->blob_by_name(P_NET_REGRESSION_BLOB_NAME).get();
		const caffe::Blob<float>* scoresBlob = pNet_->blob_by_name(P_NET_SCORE_BLOB_NAME).get();
		std::vector<Face> faces = composeFaces(regressionsBlob, scoresBlob, currentScale);
		std::vector<Face> facesNMS = FaceDetector::nonMaximumSuppression(faces, 0.8f);
		
		if (!facesNMS.empty()) {
			finalFaces.insert(finalFaces.end(), facesNMS.begin(), facesNMS.end()); 
		}
		faceSize /= scaleFactor;
	}
	finalFaces = FaceDetector::nonMaximumSuppression(finalFaces, 0.8f);
	Face::applyRegression(finalFaces, false);
	Face::bboxes2Squares(finalFaces);
	return finalFaces;
}

std::vector<Face> FaceDetector::step2(cv::Mat img, const std::vector<Face>& faces) {
	std::vector<Face> finalFaces;
	cv::Size windowSize = cv::Size(rNet_->input_blobs()[0]->width(), rNet_->input_blobs()[0]->height());
	for (size_t i = 0; i < faces.size(); ++i) {
		cv::Mat sample = cropImage(img, faces[i].bbox.getRect());
		cv::resize(sample, sample, windowSize, 0, 0, cv::INTER_AREA);
		initNetInput(rNet_, sample);
		rNet_->Forward();
		const caffe::Blob<float>* regressionBlob = rNet_->blob_by_name(R_NET_REGRESSION_BLOB_NAME).get();
		const caffe::Blob<float>* scoreBlob = rNet_->blob_by_name(R_NET_SCORE_BLOB_NAME).get();
		float score = scoreBlob->cpu_data()[1];
		if (score < rThreshold_) {
			continue;
		}
		const float* regressionData = regressionBlob->cpu_data();
		Face face = faces[i];
		face.regression[0] = regressionData[0];
		face.regression[1] = regressionData[1];
		face.regression[2] = regressionData[2];
		face.regression[3] = regressionData[3];
		face.score = score;
		finalFaces.push_back(face);
	}
	finalFaces = FaceDetector::nonMaximumSuppression(finalFaces, 0.8f);
	Face::applyRegression(finalFaces, true);
	Face::bboxes2Squares(finalFaces);
	return finalFaces;
}

std::vector<Face> FaceDetector::step3(cv::Mat img, const std::vector<Face>& faces) {
	std::vector<Face> finalFaces;
	cv::Size windowSize = cv::Size(oNet_->input_blobs()[0]->width(), oNet_->input_blobs()[0]->height());
	for (size_t i = 0; i < faces.size(); ++i) {
		cv::Mat sample = cropImage(img, faces[i].bbox.getRect());
		cv::resize(sample, sample, windowSize, 0, 0, cv::INTER_AREA);
		initNetInput(oNet_, sample);
		oNet_->Forward();
		const caffe::Blob<float>* regressionBlob = oNet_->blob_by_name(O_NET_REGRESSION_BLOB_NAME).get();
		const caffe::Blob<float>* scoreBlob = oNet_->blob_by_name(O_NET_SCORE_BLOB_NAME).get();
		const caffe::Blob<float>* ptsBlob = oNet_->blob_by_name(O_NET_PTS_BLOB_NAME).get();
		float score = scoreBlob->cpu_data()[1];
		if (score < oThreshold_) {
			continue;
		}
		const float* regressionData = regressionBlob->cpu_data();
		Face face = faces[i];
		face.regression[0] = regressionData[0];
		face.regression[1] = regressionData[1];
		face.regression[2] = regressionData[2];
		face.regression[3] = regressionData[3];
		face.score = score;
		const float* ptsData = ptsBlob->cpu_data();
		for (int p = 0; p < NUM_PTS; ++p) {
			face.ptsCoords[2 * p] = face.bbox.x1 + ptsData[p + NUM_PTS] * (face.bbox.x2 - face.bbox.x1 + 1) - 1;
			face.ptsCoords[2 * p + 1] = face.bbox.y1 + ptsData[p] * (face.bbox.y2 - face.bbox.y1 + 1) - 1;
		}
		finalFaces.push_back(face);
	}
	Face::applyRegression(finalFaces, true);
	finalFaces = FaceDetector::nonMaximumSuppression(finalFaces, 0.8f, true);
	return finalFaces;
}

std::vector<Face> FaceDetector::step4(cv::Mat img, const std::vector<Face>& faces) {
	std::vector<Face> finalFaces;
	cv::Size windowSize = cv::Size(lNet_->input_blobs()[0]->width(), lNet_->input_blobs()[0]->height());
	for (size_t i = 0; i < faces.size(); ++i) {
		std::vector<cv::Mat> samples;
		std::vector<cv::Rect> patches;
		for (int p = 0; p < NUM_PTS; ++p) {
			float maxSide = std::max(faces[i].bbox.x2 - faces[i].bbox.x1, faces[i].bbox.y2 - faces[i].bbox.y1);
			int patchSide = std::floor(0.25f *(maxSide + 1));
			if (patchSide % 2 == 1) { 
				++patchSide;
			}
			int patchX = std::floor(faces[i].ptsCoords[2 * p] - 0.5f * patchSide);
			int patchY = std::floor(faces[i].ptsCoords[2 * p + 1] - 0.5f * patchSide);
			cv::Rect patch(patchX, patchY, patchSide, patchSide);
			cv::Mat sample = cropImage(img, patch);
			cv::resize(sample, sample, windowSize, 0, 0, cv::INTER_AREA);
			samples.push_back(sample);
			patches.push_back(patch);
		}
		initNetInput(lNet_, samples);
		lNet_->Forward();
		Face face = faces[i];
		for (int p = 0; p < NUM_PTS; ++p) {
			const caffe::Blob<float>* regressionBlob = lNet_->output_blobs()[p];
			const float* regressionData = regressionBlob->cpu_data();
			float dx = regressionData[1];
			float dy = regressionData[0];
			if (std::abs(dx - 0.5f) < L_THRESHOLD && std::abs(dy - 0.5f) < L_THRESHOLD) { 
				face.ptsCoords[2 * p] += dx * patches[p].width - 0.5f * patches[p].width;
				face.ptsCoords[2 * p + 1] += dy * patches[p].height - 0.5f * patches[p].height;
			}
		}
		finalFaces.push_back(face);
	}
	return finalFaces;
}

std::vector<Face> FaceDetector::nonMaximumSuppression(std::vector<Face> faces, float threshold, bool useMin) {
	std::vector<Face> facesNMS;
	if (faces.empty()) {
		return facesNMS;
	}
	std::sort(faces.begin(), faces.end(), [](const Face& f1, const Face& f2) {
		return f1.score > f2.score;
	});
	std::vector<int> indices(faces.size());
	for (size_t i = 0; i < indices.size(); ++i) {
		indices[i] = i;
	}
	while (indices.size() > 0) {
		int idx = indices[0];
		facesNMS.push_back(faces[idx]);
		std::vector<int> tmpIndices = indices;
		indices.clear();
		for(size_t i = 1; i < tmpIndices.size(); ++i) {
			int tmpIdx = tmpIndices[i];
			float interX1 = std::max(faces[idx].bbox.x1, faces[tmpIdx].bbox.x1);
			float interY1 = std::max(faces[idx].bbox.y1, faces[tmpIdx].bbox.y1);
			float interX2 = std::min(faces[idx].bbox.x2, faces[tmpIdx].bbox.x2);
			float interY2 = std::min(faces[idx].bbox.y2, faces[tmpIdx].bbox.y2);
			 
			float bboxWidth = std::max(0.f, (interX2 - interX1 + 1));
			float bboxHeight = std::max(0.f, (interY2 - interY1 + 1));

			float interArea = bboxWidth * bboxHeight;
			// TODO: compute outside the loop
			float area1 = (faces[idx].bbox.x2 - faces[idx].bbox.x1 + 1) * 
							(faces[idx].bbox.y2 - faces[idx].bbox.y1 + 1);
			float area2 = (faces[tmpIdx].bbox.x2 - faces[tmpIdx].bbox.x1 + 1) * 
							(faces[tmpIdx].bbox.y2 - faces[tmpIdx].bbox.y1 + 1);
			float o = 0.f;
			if (useMin) {
				o = interArea / std::min(area1, area2);           
			} else {
				o = interArea / (area1 + area2 - interArea);
			}
			if(o <= threshold) {
			    indices.push_back(tmpIdx);
			}
		}
	}
	return facesNMS;
}

std::vector<Face> FaceDetector::composeFaces(const caffe::Blob<float>* regressionsBlob, 
									  const caffe::Blob<float>* scoresBlob, 
									  float scale) {
	assert(regressionsBlob->num() == 1 && scoresBlob->num() == 1);
	assert(regressionsBlob->channels() == 4 && scoresBlob->channels() == 2);
	std::vector<Face> faces;
	const int windowSide = static_cast<int>(P_NET_WINDOW_SIDE);

	const int height = regressionsBlob->height();
	const int width = regressionsBlob->width();

	const float* regressionsData = regressionsBlob->cpu_data();
	const float* scoresData = scoresBlob->cpu_data();
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			float score = scoresData[1 * width * height + y * width + x];
			if (score < pThreshold_) {
				continue;
			}
			Face face;
			face.bbox.x1 = std::floor((P_NET_STRIDE * x + 1) / scale);
			face.bbox.y1 = std::floor((P_NET_STRIDE * y + 1) / scale);
			face.bbox.x2 = std::floor((P_NET_STRIDE * x + windowSide) / scale);
			face.bbox.y2 = std::floor((P_NET_STRIDE * y + windowSide) / scale);
			face.regression[0] = regressionsData[0 * width * height + y * width + x];
			face.regression[1] = regressionsData[1 * width * height + y * width + x];
			face.regression[2] = regressionsData[2 * width * height + y * width + x];
			face.regression[3] = regressionsData[3 * width * height + y * width + x];
			face.score = score;
			faces.push_back(face);
		}
	}
	return faces;
}

cv::Rect BBox::getRect() const {
	return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

BBox BBox::getSquare() const {
	BBox bbox;
	float bboxWidth = x2 - x1;
	float bboxHeight = y2 - y1;
	float side = std::max(bboxWidth, bboxHeight);
	bbox.x1 = static_cast<int>(x1 + (bboxWidth - side) * 0.5f);
	bbox.y1 = static_cast<int>(y1 + (bboxHeight - side) * 0.5f);
	bbox.x2 = static_cast<int>(bbox.x1 + side);
	bbox.y2 = static_cast<int>(bbox.y1 + side);
	return bbox;
}

void Face::applyRegression(std::vector<Face>& faces, bool addOne) {
	for (size_t i = 0; i < faces.size(); ++i) {
		float bboxWidth = faces[i].bbox.x2 - faces[i].bbox.x1 + static_cast<float>(addOne);
		float bboxHeight = faces[i].bbox.y2 - faces[i].bbox.y1 + static_cast<float>(addOne);
		faces[i].bbox.x1 = faces[i].bbox.x1 + faces[i].regression[1] * bboxWidth;
		faces[i].bbox.y1 = faces[i].bbox.y1 + faces[i].regression[0] * bboxHeight;
		faces[i].bbox.x2 = faces[i].bbox.x2 + faces[i].regression[3] * bboxWidth;
		faces[i].bbox.y2 = faces[i].bbox.y2 + faces[i].regression[2] * bboxHeight;
	}
}

void Face::bboxes2Squares(std::vector<Face>& faces) {
	for (size_t i = 0; i < faces.size(); ++i) {
		faces[i].bbox = faces[i].bbox.getSquare();
	}
}

}
