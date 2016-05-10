#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string>
#include<conio.h>
#include <opencv/cv.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <fstream>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>



using namespace cv;
using namespace std;

// Define Constants
const int areathresh = 50;//75
const int noofruns = 35;//35
const int disti = 5;
const int hood = 3; // checking for edge points in a (hood X hood) area
const float stdgaus = 19.8843;
const float meangaus = 96.8514;
const int psudoB = 0;
const int psudoG = 128;
const int psudoR = 255;

void coverup(Mat  & blackimage)
{
	Rect cover = Rect(750, 0, 250, 224); // coordinates ( 807,0) (807,224), (1023, 0), (1023, 224)
	Mat roi_cover = blackimage(cover);
	roi_cover = cv::Scalar(104, 40, 14);
	imwrite("tempm.png", blackimage);

}

void removebackground(Mat colorbarimg)
{
	Mat orgim = colorbarimg; //Load source image

	for (int x = 0; x < colorbarimg.rows; x++)
	{
		for (int y = 0; y < colorbarimg.cols; y++)
		{
			if (x == 0 || y == 0 || x == colorbarimg.rows-1 || y == colorbarimg.cols-1)
			{ }
			else
			{
				float avinten = ((colorbarimg.at<cv::Vec3b>(y, x)[0] + colorbarimg.at<cv::Vec3b>(y, x)[1] + colorbarimg.at<cv::Vec3b>(y, x)[2])
					+ (colorbarimg.at<cv::Vec3b>(y, x - 1)[0] + colorbarimg.at<cv::Vec3b>(y, x - 1)[1] + colorbarimg.at<cv::Vec3b>(y, x - 1)[2])
					+ (colorbarimg.at<cv::Vec3b>(y, x + 1)[0] + colorbarimg.at<cv::Vec3b>(y, x + 1)[1] + colorbarimg.at<cv::Vec3b>(y, x + 1)[2])
					+ (colorbarimg.at<cv::Vec3b>(y - 1, x - 1)[0] + colorbarimg.at<cv::Vec3b>(y - 1, x - 1)[1] + colorbarimg.at<cv::Vec3b>(y - 1, x - 1)[2])
					+ (colorbarimg.at<cv::Vec3b>(y - 1, x)[0] + colorbarimg.at<cv::Vec3b>(y - 1, x)[1] + colorbarimg.at<cv::Vec3b>(y - 1, x)[2])
					+ (colorbarimg.at<cv::Vec3b>(y - 1, x + 1)[0] + colorbarimg.at<cv::Vec3b>(y - 1, x + 1)[1] + colorbarimg.at<cv::Vec3b>(y - 1, x + 1)[2])
					+ (colorbarimg.at<cv::Vec3b>(y + 1, x - 1)[0] + colorbarimg.at<cv::Vec3b>(y + 1, x - 1)[1] + colorbarimg.at<cv::Vec3b>(y + 1, x - 1)[2])
					+ (colorbarimg.at<cv::Vec3b>(y + 1, x)[0] + colorbarimg.at<cv::Vec3b>(y + 1, x)[1] + colorbarimg.at<cv::Vec3b>(y + 1, x)[2])
					+ (colorbarimg.at<cv::Vec3b>(y + 1, x + 1)[0] + colorbarimg.at<cv::Vec3b>(y + 1, x + 1)[1] + colorbarimg.at<cv::Vec3b>(y + 1, x + 1)[2])) / 9;

				float exppo = -(pow((avinten - meangaus), 2)) / (2 * stdgaus*stdgaus);
				float pro = (1 / (stdgaus*sqrt(2 * 3.14)))*exp(exppo);
				
				if (pro > 0.01)
				{
					//cout << "hoo" << endl;
					orgim.at<cv::Vec3b>(y, x)[0] = psudoB; orgim.at<cv::Vec3b>(y, x)[1] = psudoG; orgim.at<cv::Vec3b>(y, x)[2] = psudoR;
					orgim.at<cv::Vec3b>(y, x - 1)[0] = psudoB; orgim.at<cv::Vec3b>(y, x - 1)[1] = psudoG; orgim.at<cv::Vec3b>(y, x - 1)[2] = psudoR;
					orgim.at<cv::Vec3b>(y, x + 1)[0] = psudoB; orgim.at<cv::Vec3b>(y, x + 1)[1] = psudoG; orgim.at<cv::Vec3b>(y, x + 1)[2] = psudoR;
					orgim.at<cv::Vec3b>(y - 1, x - 1)[0] = psudoB; orgim.at<cv::Vec3b>(y - 1, x - 1)[1] = psudoG; orgim.at<cv::Vec3b>(y - 1, x - 1)[2] = psudoR;
					orgim.at<cv::Vec3b>(y - 1, x)[0] = psudoB; orgim.at<cv::Vec3b>(y - 1, x)[1] = psudoG; orgim.at<cv::Vec3b>(y - 1, x)[2] = psudoR;
					orgim.at<cv::Vec3b>(y - 1, x + 1)[0] = psudoB; orgim.at<cv::Vec3b>(y - 1, x + 1)[1] = psudoG; orgim.at<cv::Vec3b>(y - 1, x + 1)[2] = psudoR;
					orgim.at<cv::Vec3b>(y + 1, x - 1)[0] = psudoB; orgim.at<cv::Vec3b>(y + 1, x - 1)[1] = psudoG; orgim.at<cv::Vec3b>(y + 1, x - 1)[2] = psudoR;
					orgim.at<cv::Vec3b>(y + 1, x)[0] = psudoB; orgim.at<cv::Vec3b>(y + 1, x)[1] = psudoG; orgim.at<cv::Vec3b>(y + 1, x)[2] = psudoR;
					orgim.at<cv::Vec3b>(y + 1, x + 1)[0] = psudoB; orgim.at<cv::Vec3b>(y + 1, x + 1)[1] = psudoG; orgim.at<cv::Vec3b>(y + 1, x + 1)[2] = psudoR;
				}

			}
		}
	}
	imshow(" psudocolored", orgim);
	imwrite("psudocolored.png", orgim);
}

Mat findedges(Mat blackimage)
{
	Mat grayimage; int scale = 1; int delta = 0;
	//int ddepth = CV_16S;
	Mat abs_grad_x, abs_grad_y, grad_x, grad_y, grad;
	GaussianBlur(blackimage, blackimage, Size(9, 9), 0, 0, 4);  // removing unnecessary noisy background
	cvtColor(blackimage, grayimage, CV_BGR2GRAY);
	//imshow("gray", grayimage);
	Sobel(grayimage, grad_x, -1, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(grayimage, grad_y, -1, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	/// Total Gradient (approximate)
	//grad = abs_grad_x + abs_grad_y;
	Mat cm_img0;
	addWeighted(abs_grad_x, 2, abs_grad_y, 2, 0, grad);
	applyColorMap(grad, cm_img0, COLORMAP_HOT);

	cv::inRange(cm_img0, cv::Scalar(250,250,250), cv::Scalar(255, 255, 255), grad);
	imshow("edges", grad); imwrite("edges.png", grad);
	
	// Apply the colormap:
	
	// Show the result:
	imshow("cm_img0", cm_img0);
	imwrite("colormap.png", cm_img0);
	removebackground(cm_img0);

	return grad;
}



// =======================================FUNCTION: ASSIGNPROB assign probability based on if a color is a boundary color=====================================================
float assignprob(Mat  blackimage, int x, int y)
{

	Mat labim; float prob; double delta;
	// Benchmarks for R, G, and B in terms of LAB color space
	Scalar benchg(44, 233, 34);  // benchmark for green (44, 233, 34);
	Scalar benchr(31, 19, 183);         // benchmark for red (31, 19, 183); 
	Scalar benchw(252, 254, 245);  // benchmark for white (252, 254, 245);
	Scalar benchlg(209, 173, 35);
	//cvtColor(blackimage, labim, CV_BGR2Lab);//conversion to lab color space
	double Lim = blackimage.at<cv::Vec3b>(y, x)[0];
	double Aim = blackimage.at<cv::Vec3b>(y, x)[1];
	double Bim = blackimage.at<cv::Vec3b>(y, x)[2];



	double deltaw = cv::sqrt(((Lim - benchw[0])*(Lim - benchw[0])) + ((Aim - benchw[1])*(Aim - benchw[1])) + ((Bim - benchw[2])*(Bim - benchw[2])));
	deltaw = cv::abs((Lim - benchw[0]) + (Aim - benchw[1]) + (Bim - benchw[2]));
	//deltaw = 1 - exp(deltaw);
	double deltar = cv::sqrt(((Lim - benchr[0])*(Lim - benchr[0])) + ((Aim - benchr[1])*(Aim - benchr[1])) + ((Bim - benchr[2])*(Bim - benchr[2])));
	deltar = cv::abs((Lim - benchr[0]) + (Aim - benchr[1]) + (Bim - benchr[2]));
	//deltar = 1 - exp(deltar);
	double deltag = cv::sqrt(((Lim - benchg[0])*(Lim - benchg[0])) + ((Aim - benchg[1])*(Aim - benchg[1])) + ((Bim - benchg[2])*(Bim - benchg[2])));
	deltag = cv::abs((Lim - benchg[0]) + (Aim - benchg[1]) + (Bim - benchg[2]));
	double deltalg = cv::abs((Lim - benchg[0]) + (Aim - benchg[1]) + (Bim - benchg[2]));
	//deltag = 1 - exp(deltag);
	// Defining Probabilities based on Delta E values


	if (deltag <= 60 || deltar <= 60 || deltaw <= 60 || deltalg <= 60) //Not perceptible by human eyes.
		prob = 1;
	else if (deltag > 60 && deltag <= 120 || deltar > 60 && deltar <= 120 || deltaw > 60 && deltaw <= 120 || deltalg > 60 && deltalg <= 120)// Perceptible through close observation.// between 0.9 to 0.8
		prob = 0.8;
	else if (deltag > 120 && deltag <= 150 || deltar > 120 && deltar <= 150 || deltaw > 120 && deltaw <= 150 || deltalg > 120 && deltalg <= 150)//Perceptible at a glance./ between 0.8 to 0.6
		prob = 0.6;
	else if (deltag > 150 && deltag <= 200 || deltar > 150 && deltar <= 200 || deltaw > 150 && deltaw <= 200 || deltalg > 150 && deltalg <= 200)//Colors are more similar than opposite
		prob = 0.4;
	//else if (deltag > 49.0 && deltag <= 100.0 || deltar > 49.0 && deltar <= 100.0 || deltaw > 49.0 && deltaw <= 100.0)//Colors are exact opposite
	else
		prob = 0.0;

	//cout << deltag << "," << deltar << "," << deltaw << "," << prob << endl; cvWaitKey(0);
	return prob;
}

int checkbackground(Mat  blackimage, int x, int y, float probinfo, float gradpo)
{

	double Lim = blackimage.at<cv::Vec3b>(y, x)[0];
	double Aim = blackimage.at<cv::Vec3b>(y, x)[1];
	double Bim = blackimage.at<cv::Vec3b>(y, x)[2];
	Scalar benchb(90, 30, 20);// benchmark for blue background
	cv::Mat patch;

	cv::getRectSubPix(blackimage, cv::Size(3, 3), Point(x, y), patch);// get a 3*3 patch
	Scalar mean;
	mean = cv::mean(patch);
	//std::cout << mean[0] << "," << mean[1] << "," << mean[2] <<  std::endl; //blue mean

	gradpo = gradpo / 256;
	double factor = (8 * probinfo) + (2 * gradpo);
	//cout << factor << endl;
	//cout << probinfo<<","<<gradpo<<","<< factor<<endl;



	double deltab = cv::sqrt(((Lim - benchb[0])*(Lim - benchb[0])) + ((Aim - benchb[1])*(Aim - benchb[1])) + ((Bim - benchb[2])*(Bim - benchb[2])));
	deltab = cv::abs((Lim - benchb[0]) + (Aim - benchb[1]) + (Bim - benchb[2]));


	if ((mean[0] >= 90 && mean[0] <= 115) && (mean[1] >= 20 && mean[1] <= 35) && (mean[2] >= 15 && mean[2] <= 25) || (factor<5.5) || (deltab <40))//5.5
		return 1;

}


int checkstrayback(Mat  blackimage, int x, int y)
{

	double Lim = blackimage.at<cv::Vec3b>(y, x)[0];
	double Aim = blackimage.at<cv::Vec3b>(y, x)[1];
	double Bim = blackimage.at<cv::Vec3b>(y, x)[2];
	Scalar benchb(90, 22, 20);// benchmark for blue background
	cv::Mat patch;

	cv::getRectSubPix(blackimage, cv::Size(3, 3), Point(x, y), patch);// get a 3*3 patch
	Scalar mean;
	mean = cv::mean(patch);
	double deltab = cv::abs((Lim - benchb[0]) + (Aim - benchb[1]) + (Bim - benchb[2]));


	if ((mean[0] >= 90 && mean[0] <= 115) && (mean[1] >= 20 && mean[1] <= 35) && (mean[2] >= 15 && mean[2] <= 30) || (deltab <40))
		return 1;

}

//================================================FUNCTION: Finds Gradient of the Image======================================================

void findgradient(Mat blackimage, Mat & grad_x, Mat & grad_y, Mat & grad)
{
	GaussianBlur(blackimage, blackimage, Size(3, 3), 0, 0, BORDER_DEFAULT);
	/// Convert it to gray
	Mat src_gray;
	cvtColor(blackimage, src_gray, CV_RGB2GRAY);

	int scale = 1;
	int delta = 0;
	//int ddepth = CV_16S;


	/// Generate grad_x and grad_y
	Mat abs_grad_x, abs_grad_y;


	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	//void Sobel(InputArray src, OutputArray dst, int ddepth, int dx, int dy, int ksize=3, double scale=1, double delta=0, int borderType=BORDER_DEFAULT)
	Sobel(src_gray, grad_x, -1, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(src_gray, grad_y, -1, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

}


//======================================FUNCTION: Finds Gradient at each point====================================
void gradientatpoint(Mat blackimage, Mat grad, float x, float y, float &gradpo)
{


	Scalar gradval = grad.at<uchar>(Point(y, x));
	gradpo = gradval[0];

	/*Scalar gradxval = grad_x.at<uchar>(Point(y, x));
	Scalar gradyval = grad_y.at<uchar>(Point(y, x));*/
	/*Mat gradient_angle_degrees;
	bool angleInDegrees = false;
	cv::phase(gradxval[0], gradyval[0], gradient_angle_degrees, angleInDegrees);
	//imshow("GRADIENT", gradient_angle_degrees);
	//angpo = fastAtan2(gradxval[0], gradyval[0]);

	Mat popp = gradient_angle_degrees;
	angpo =popp.at<double>(0, 0);*/


	//Mat gradinfo = (gradval[0], gradient_angle_degrees);
	//cout << gradval[0]<<", "<<gradient_angle_degrees<<endl;// gradient value and angle


}


//================================================FUNCTION:Define edge points for a contour based on condition============================================================
int defineedgepoints(float probinfo, float gradpo)
{
	gradpo = gradpo / 256;
	double factor = (5 * probinfo) + (5 * gradpo);
	//cout << probinfo<<","<<gradpo<<","<< factor<<endl;
	if (factor > 4.5)
		return 1;
	else
		return 0;
}

//======================================FUNCTION: Finds Centroid of each contour==========================================================================================
vector<Point2f> findcentroid(vector<vector<Point>> contours)
{

	vector<Moments> mu(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i], false);
	}


	///  Get the mass centers:
	vector<Point2f> centroi(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		centroi[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}


	return centroi;
}

void jointhem(vector<Point> first, vector<Point> second, vector<Point> & result)
{
	for (int i = 0; i < first.size(); i++)
		result.push_back(first[i]); // pushing all first contour's points into result

	for (int j = 0; j < second.size(); j++)
	{
		if (pointPolygonTest(Mat(first), second[j], true) < 0)// not inside the contour
			result.push_back(second[j]);
	}

}

//======================================FUNCTION: Calculates angles at each point on the contour====================================
double angletime(float px2, float py2, float px1, float py1, float cx1, float cy1)
{
	double angle1 = atan2(py2 - cy1, px2 - cx1) * 180.0 / CV_PI;
	double angle2 = 0;  //double angle2 = atan2(py1 - cy1, px1 - cx1) * 180.0 / CV_PI;
	double ang = angle1 - angle2;

	if (ang > 360)
		ang = ang - 360;
	else if (ang < 0)
		ang = ang + 360;
	else
		ang = ang;

	ang = ang* CV_PI / 180;// converting to radians
	//cout << ang << endl;
	return ang;// returns angle at a single point on the contour
}


//================================================FUNCTION: Go to the new point============================================================
void gotonewpoint(int x, int y, int & xpo, int & ypo, double theta, int dist)// takes in point(x,y)and angle to move in
{



	//theta = CV_PI;
	if (x<26 || y<26 || x>1000 || y>1000)
	{
		xpo = x; ypo = y;
	}
	else
	{
		xpo = x + dist*cos(theta);// angle must be specified in radians
		ypo = y + dist*sin(theta);
	}



	//xpo= (int)round(x + dist * cos(theta * CV_PI / 180.0));
	//ypo = (int)round(y + dist * sin(theta * CV_PI / 180.0));//angle in degrees
}

vector<vector<Point>> removestrayblack(Mat blackimage, vector<vector<Point>> contours, vector<Point2f> centroi)
{
	int xpo = 0, ypo = 0; vector<vector<Point>> bon; vector<vector<Point>> gon;

	for (int id = 0; id < contours.size(); id++) // going through all contours
	{
		int hop = 0;
		for (int po = 0; po < contours[id].size(); po++)
		{
			double A = angletime(contours[id][po].x, contours[id][po].y, centroi[id].x + 15, centroi[id].y, centroi[id].x, centroi[id].y);// finding angle at each point of the contour
			gotonewpoint(contours[id][po].x, contours[id][po].y, xpo, ypo, A, disti);
			int isback = checkstrayback(blackimage, xpo, ypo);
			if (isback == 1)
				hop++;
		}

		if (hop > contours[id].size() / 4)// it is a stray black
			bon.push_back(contours[id]);
		else
			gon.push_back(contours[id]);


	}
	for (int uu = 0; uu < gon.size(); uu++)
		drawContours(blackimage, gon, uu, Scalar(0, 255, 0), 2, 8, vector<Vec4i>(), 0, Point());

	//imshow("Stray", blackimage);
	return gon;
}


vector<vector<Point>> joincontours(vector<vector<Point>> DrawCont, Mat blackimage)
{
	int yop = 0;
	Mat img1(blackimage.rows, blackimage.cols, CV_8UC1);
	Mat img2(blackimage.rows, blackimage.cols, CV_8UC1);
	vector<Point2f>center(DrawCont.size());
	vector<float>radius(DrawCont.size());
	vector<vector<Point>> temp;
	Mat res; vector<int> pint; vector<Point> savv;
	// checking to see if points inside a particular contour is part of a other contour
	/*for (int ho = 0; ho < DrawCont.size(); ho++)
	minEnclosingCircle((Mat)DrawCont[ho], center[ho], radius[ho]);*/
	//float dist = cv::sqrt(((center[ho].x - center[j].x)*(center[ho].x - center[j].x)) + ((center[ho].y - center[j].y)*(center[ho].y - center[j].y)));

	for (int ho = 0; ho <DrawCont.size(); ho++)
	{
		img1.setTo(Scalar(255, 255, 255));
		drawContours(img1, DrawCont, ho, Scalar(0, 0, 0), 2, 8, vector<Vec4i>(), 0, Point());
		for (int j = 0; j < DrawCont.size(); j++)

		{
			img2.setTo(Scalar(255, 255, 255));
			drawContours(img2, DrawCont, j, Scalar(0, 0, 0), 2, 8, vector<Vec4i>(), 0, Point());
			yop = 0;
			bitwise_and(img1, img2, res);
			for (int iu = 0; iu < res.rows; iu++) //checking through the image
			{
				for (int ju = 0; ju < res.cols; ju++)
				{
					if (res.at<uchar>(ju, iu) == 1)
						yop++;
				}

			}
			//cout << yop<<endl;
			//imshow("img1", img2);
			//if (dist<30)

			if (yop > 0)
				cout << "hello" << endl;;
			jointhem(DrawCont[j], DrawCont[ho], savv);
			temp.push_back(savv);
			pint.push_back(j);
			pint.push_back(ho);
		}
	}

	//imshow("img1", res);
	for (int t = 0; t < DrawCont.size() / 2; t++)
	{
		for (int tu = 0; tu < pint.size(); tu++)
		{
			if (t != tu)
			{
				temp.push_back(DrawCont[t]);
			}
		}
	}


	for (int id = 0; id < temp.size(); id++)
		drawContours(blackimage, temp, id, Scalar(0, 255, 0), 2, 8, vector<Vec4i>(), 0, Point());

	imshow("joined", blackimage);
	return temp;



}


vector<vector<Point>> movesomemore(vector<vector<Point>> DrawCont, Mat blackimage)
{
	vector<Point2f> centro = findcentroid(DrawCont); int xpo, ypo; vector<Point> cont; vector<vector<Point>> contsto;
	for (int id = 0; id < DrawCont.size(); id++)
	{
		for (int pd = 0; pd < DrawCont[id].size(); pd++)
		{
			double A = angletime(DrawCont[id][pd].x, DrawCont[id][pd].y, centro[id].x + 15, centro[id].y, centro[id].x, centro[id].y);// finding angle at each point of the contour

			gotonewpoint(DrawCont[id][pd].x, DrawCont[id][pd].y, xpo, ypo, A, 7);
			cont.push_back(Point(xpo, ypo));
		}
		contsto.push_back(cont);
		cont.clear();
	}
	return contsto;
}

void showcontour(vector<vector<Point>> contours, Mat blackimage)
{
	Scalar benchw(252, 254, 245);  // benchmark for white (252, 254, 245); 
	vector<vector<Point>> coon; vector<vector<Point>> fin; vector<Point> finp;
	for (int id = 0; id < contours.size(); id++) // going through all contours and saving the ones greater than are 15000
	{
		float chaid = contourArea(contours[id], false);
		if (chaid > 15000)
			coon.push_back(contours[id]);
	}

	vector<Point2f> centroi = findcentroid(coon);// finding centroids of these large contours
	int xpo = 0, ypo = 0;
	for (int yu = 0; yu < coon.size(); yu++)
	{

		for (int in = 0; in < coon[yu].size(); in++)
		{

			double Lim = blackimage.at<cv::Vec3b>(coon[yu][in].y, coon[yu][in].x)[0];
			double Aim = blackimage.at<cv::Vec3b>(coon[yu][in].y, coon[yu][in].x)[1];
			double Bim = blackimage.at<cv::Vec3b>(coon[yu][in].y, coon[yu][in].x)[2];
			double deltaw = cv::abs((Lim - benchw[0]) + (Aim - benchw[1]) + (Bim - benchw[2]));
			xpo = coon[yu][in].x; ypo = coon[yu][in].y;
		HOP:                if (deltaw > 120)
		{
			float A = angletime(coon[yu][in].x, coon[yu][in].y, centroi[yu].x + 15, centroi[yu].y, centroi[yu].x, centroi[yu].y);// finding angle at each point of the contour
			gotonewpoint(coon[yu][in].x, coon[yu][in].y, xpo, ypo, A, 5);

		}
							if (deltaw < 120)
								finp.push_back(Point(xpo, ypo));
							else
								goto HOP;


		}
		fin.push_back(finp);
	}

	for (int id = 0; id < fin.size(); id++)
		drawContours(blackimage, fin, id, Scalar(0, 255, 0), 2, 8, vector<Vec4i>(), 0, Point());


	imshow("Huge", blackimage);
}
int howmanyedgepoints(Mat edges, int x, int y)
{
	int edgpoints = 0;
	if ((x>10) && (x<1010) && (y>10) && (y < 1010))
	{
		for (int i = x - hood; i <= x + hood; i++)
		{
			for (int j = y - hood; j <= y + hood; j++)
			{
				if (edges.at<uchar>(j, i) == 0) //checking presense of edge points
					edgpoints++;
			}
		}
	}
	if (edgpoints > 8)
		return 1;
	else
		return 0;
}



vector<vector<Point>> greencontourdetect()
{
	//(%%%%%%%%%%%%%%%%%%%%%%        Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%)
	vector<vector<Point>> whitogcontours; // Vector for storing contour
	vector<vector<Point>> anocontours;
	vector<Vec4i> whitoghierarchy;
	Scalar color(255, 255, 255);
	char sizee;
	Mat extr; Mat extro;
	Mat blackimage = imread("tempm.png"); //Load source image
	Mat blackimage1 = imread("tempm.png");
	/// Do the operation new_image(i,j) = alpha*image(i,j) + beta

	Mat mask = cv::Mat(blackimage1.rows, blackimage1.cols, CV_8UC1);
	Mat thr_blackimage(blackimage.rows, blackimage.cols, CV_8UC1);
	Mat blackdst(blackimage.rows, blackimage.cols, CV_8UC1, Scalar::all(0));


	//(%%%%%%%%%%%%%%%%%%%%%%        Thresholding %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%)

	//cv::inRange(blackimage, cv::Scalar(86, 155, 0), cv::Scalar(255, 255, 150), thr_blackimage);//GREEN 86,155,0  
	cv::inRange(blackimage, cv::Scalar(160, 80, 0), cv::Scalar(230, 200, 20), thr_blackimage);//BLUE GREEN 86,155,0  
	blur(thr_blackimage, thr_blackimage, Size(2, 2));
	dilate(thr_blackimage, thr_blackimage, Mat());
	//imshow("blue", thr_blackimage);

	cv::Mat const struc = cv::getStructuringElement(
		cv::MORPH_RECT, cv::Size(9, 7));
	cv::morphologyEx(thr_blackimage, thr_blackimage, cv::MORPH_CLOSE, struc);

	//cv::inRange(blackimage, cv::Scalar(75, 20, 110), cv::Scalar(255, 128, 255), extr);
	//imshow("Green theresholded image", thr_blackimage);
	cv::inRange(blackimage, cv::Scalar(0, 20, 110), cv::Scalar(185, 113, 255), extro);//(185,113,255)//REd
	//cv::inRange(blackimage, cv::Scalar(80, 10, 105), cv::Scalar(200, 90, 200), extro);//bright red
	//imshow("Red Thresholded Image", extro);
	//dilate(extr, extr, Mat());
	//dilate(extr, extr, Mat());
	Mat dub;
	vector<Point> imcon;
	//cv::inRange(blackimage, cv::Scalar(140,140,140), cv::Scalar(255,255, 255), dub);//WHITE
	cv::inRange(blackimage, cv::Scalar(200, 200, 200), cv::Scalar(255, 255, 255), dub);
	Mat summy = thr_blackimage + dub + extro;
	dilate(summy, summy, Mat());
	dilate(summy, summy, Mat());
	erode(summy, summy, Mat());
	//imshow("summy", summy);

	//blur(summy, summy, Size(3,3));
	cv::Mat const structure_elem = cv::getStructuringElement(
		cv::MORPH_RECT, cv::Size(7, 7));
	cv::Mat close_result;
	cv::morphologyEx(summy, summy, cv::MORPH_CLOSE, structure_elem);
	//imshow("Green+Red", summy);
	//(%%%%%%%%%%%%%%%%%%%%%%        Finiding Contours %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%)
	findContours(summy, whitogcontours, whitoghierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); // Find the contours in the image


	return whitogcontours;



}


vector<Point> newcontours(Mat edges, vector<Point> oldcont, Point2f centro)
{
	vector<Point> newcont; int checkx = 0, checky = 0; int dis;
	newcont.clear();
	for (int i = 0; i < oldcont.size(); i++)
	{
		int nedges = howmanyedgepoints(edges, oldcont[i].x, oldcont[i].y);
		float ara = contourArea(oldcont, false);

		if (ara > 2000)
			dis = 10;
		else
			dis = 5;

		if (nedges> 15)//  if the point in consideration is an edge point
			newcont.push_back(Point(oldcont[i].x, oldcont[i].y));

		else    // if the point in consideration is not an edge point
		{

			double A = angletime(oldcont[i].x, oldcont[i].y, centro.x + 15, centro.y, centro.x, centro.y);// finding angle at each point of the contour
			gotonewpoint(oldcont[i].x, oldcont[i].y, checkx, checky, A, dis);
			int nedg = howmanyedgepoints(edges, checkx, checky);
			newcont.push_back(Point(checkx, checky));


		}
	}
	return newcont;
}


void bronchannotate(vector<vector<Point>> blcont, Mat & blackimage1)
{
	vector<Rect> boundRect(blcont.size());
	vector<Vec4i> convexityDefects;
	vector<vector<Point> >hullp(blcont.size());
	vector<vector<int> >hull(blcont.size());
	for (int i = 0; i < blcont.size(); i++)
	{
		convexHull(Mat(blcont[i]), hull[i], false);
		cv::convexityDefects(blcont[i], hull[i], convexityDefects);
		boundRect[i] = boundingRect(Mat(blcont[i]));
		float ar = contourArea(blcont[i], false);
		if (ar > 15000)
		{
			if (convexityDefects.size() > 18)
				putText(blackimage1, "Artery", Point(boundRect[i].x, boundRect[i].y + 15), FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(255, 255, 0), 1, CV_AA);
			else
				putText(blackimage1, "Central Bronch", Point(boundRect[i].x, boundRect[i].y + 15), FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(255, 255, 0), 1, CV_AA);
		}
	}

}

void otherannotate(vector<vector<Point>> mcontours, vector<vector<Point>> lapo, Mat & blackimage1)
{
	vector<Rect> boundRect(mcontours.size());
	Mat oo = imread("tempm.png"); Mat dub;
	cv::inRange(oo, cv::Scalar(200, 200, 200), cv::Scalar(255, 255, 255), dub);
	imshow("wjite", dub);
	for (int i = 0; i < mcontours.size(); i++) // going through each contour
	{
		float chid = mcontours.size();
		boundRect[i] = boundingRect(Mat(mcontours[i]));
		Rect mow = boundRect[i];
		Mat wow = dub(mow);
		if (chid < 350)
		{
			int yap = countNonZero(wow);
			if (yap> 10)
				putText(blackimage1, "TBr", Point(boundRect[i].x, boundRect[i].y + 15), FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(255, 255, 0), 1, CV_AA);
			else
				putText(blackimage1, "AT/AB", Point(boundRect[i].x, boundRect[i].y + 15), FONT_HERSHEY_COMPLEX_SMALL, 0.8, CV_RGB(255, 255, 0), 1, CV_AA);
		}
	}
}

//===================================================MAIN STARTS==========================================================

int main()
{
	// =============================== Initialization===============================================

	Mat blackimage = imread("16.5/16.5.1/2014-012-019_C57Bl6_LMM.14.24.4.34_E16.5_SOX9_NKX2.1_ACTA2-04.png"); //Load source image
	//GaussianBlur(blackimage, blackimage, Size(3, 3), 0, 0, BORDER_DEFAULT);
	float gradpo; double angpo; int geo = 0; int topo = 0; int cun = 0; int diout = 5;
	Mat checkop; Mat grad_x, grad_y, grad; int top = 0;
	int largest_contour_index = 0; double A;
	vector<vector<Point>> acontours;
	vector<vector<Point>> contours; // Vector for storing contour
	vector<Point>storag; vector<vector<Point>> storconts;
	vector<Vec4i> hierarchy; int conty = 0; int edpoint;
	int dec; Point pin; float probinfo; double aid;
	vector<vector<Point>> DrawCont; vector<Point>coont; int xpo = 0; int ypo = 0;
	vector<vector<double>> angstore; vector<double> ango;// to associate angles with each point
	int isback; int bathresh = 0; int nopoint; vector<Point> backup; vector<int> ido; int did = 0;
	/*~~~~~CALL~~~~~~*/coverup(blackimage);//Covers up the written text
	Mat blackimage1 = imread("tempm.png");
	Mat blackimage2 = imread("tempm.png");
	Mat untouched = imread("tempm.png");
	Mat img1(blackimage.rows, blackimage.cols, CV_8UC1);
	// =============================== =================================================================

	Mat edges = findedges(blackimage2);
	/*~~~~~CALL~~~~~~*/findgradient(blackimage1, grad_x, grad_y, grad);// finds gradient of the whole image once in the program
	// =============================== Find Contours===============================================
	//===== ======================= Thresholding &  Contouring===============================================
	Mat thr_blackimage(blackimage.rows, blackimage.cols, CV_8UC1);
	GaussianBlur(blackimage, blackimage, Size(7, 7), 0, 0, 4);  // removing unnecessary noisy background
	cv::inRange(blackimage, cv::Scalar(0, 0, 0), cv::Scalar(35, 45, 45), thr_blackimage);//35, 30, 30)  35, 45, 35
	//dilate(thr_blackimage, thr_blackimage, Mat());
	//imshow("check", thr_blackimage);

	findContours(thr_blackimage, acontours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // find all black contours

	for (int dop = 0; dop < acontours.size(); dop++)  // finding and saving the contours that are above a certain area
	{
		float chaid = contourArea(acontours[dop], false);

		if (chaid > areathresh)
		{
			contours.push_back(acontours[dop]); //saving only the large contours in another variable

		}
	}
	//bronchannotate(contours, blackimage1);
	for (int d = 0; d < contours.size(); d++)  // finding and saving the contours that are above a certain area
	{
		drawContours(blackimage2, contours, d, Scalar(0, 255, 0), 2, 8, vector<Vec4i>(), 0, Point());
	}
	imshow("Black contours", blackimage2);

	/*~~~~~CALL~~~~~~*/vector<Point2f> centroi = findcentroid(contours);// finding centroid of each relaevant contour once for the image

	///*~~~~~CALL~~~~~~*/contours = removestrayblack(blackimage, contours, centroi);
	centroi = findcentroid(contours);// finding centroid of each relaevant contour once for the image
	//wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww  WHILE LOOP  wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww

	//joinnagain(DrawCont, untouched); //while ends
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ DRAW CONTOURS, IMSHOW & END ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	vector<vector<Point>> tempacontours(contours.size()); vector<vector<Point>> mcontours(tempacontours.size());


	for (int i = 0; i < contours.size(); i++) // going through each contour
	{
		float chid = contourArea(contours[i], false);
		cun = 0;//count of number of edges


		//FIRST RUN : Moving the points in the contour by 5 and checking if edge point

		for (int j = 0; j < contours[i].size(); j++)// going through points in every contour
		{
			pin = contours[i][j];// a particular point j in contour i
			A = angletime(pin.x, pin.y, centroi[i].x + 15, centroi[i].y, centroi[i].x, centroi[i].y);// finding angle at each point of the contour
			gotonewpoint(pin.x, pin.y, xpo, ypo, A, diout);
			tempacontours[i].push_back(Point(xpo, ypo));// storing the newly moved to points into empty tempacontours
			int uu = howmanyedgepoints(edges, xpo, ypo);
			if (uu == 1)
				cun++;
		}


		// SUBSEQUENT RUNS : Moving the points in the contour by 5 and checking if edge point till atleast half the points in the contour are edge points

		if (cun < contours[i].size() / 2) // if (no of edge points in the contour < half the points in the contour)--> go for another run
		{
			cun = 0;
		pop:
			for (int jo = 0; jo < tempacontours[i].size(); jo++)// going through points in every contour
			{
				pin = tempacontours[i][jo];// a particular point j in contour 
				//pin = mcontours[i][jo];
				A = angletime(pin.x, pin.y, centroi[i].x + 15, centroi[i].y, centroi[i].x, centroi[i].y);// finding angle at each point of the contour
				gotonewpoint(pin.x, pin.y, xpo, ypo, A, diout);
				mcontours[i].push_back(Point(xpo, ypo)); // new point now put into mcontours
				if (howmanyedgepoints(edges, xpo, ypo) == 1)
					cun++;
			}
			//if not mcontours[i] has the final points we need for the ith contour
			if (cun < mcontours[i].size() / 2)
			{
				cun = 0; tempacontours[i].clear();
				for (int ya = 0; ya < tempacontours[i].size(); ya++)
					tempacontours[ya].push_back(Point(mcontours[i][ya].x, mcontours[i][ya].y));
				mcontours[i].clear();
				goto pop;
			}
		}

		else
			mcontours.push_back(tempacontours[i]);
	}


	vector<vector<Point>>lapo = movesomemore(mcontours, blackimage1);

	//otherannotate(mcontours, lapo, blackimage1);
	//===================================Display Image with drawn contours====================================================

	/*vector<vector<Point>> greencont = greencontourdetect();
	vector<vector<Point>> grcont;
	vector<vector<Point> >hullg(greencont.size());

	for (int ig = 0; ig < greencont.size(); ig++)
	{
	double ag = contourArea(greencont[ig], false);
	if (ag > 300 && ag < 1000)
	{
	convexHull(greencont[ig], hullg[ig], false);
	grcont.push_back(hullg[ig]);
	}
	}


	for (int ido = 0; ido < grcont.size(); ido++)
	{
	Scalar colo = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	//if (DrawCont[id].size()>15)// if greater than 15 points draw contours
	drawContours(blackimage1, grcont, ido, Scalar(255, 255, 125), 2, 8, vector<Vec4i>(), 0, Point());
	}*/    //green contours

	for (int ido = 0; ido < lapo.size(); ido++)
	{
		drawContours(blackimage1, lapo, ido, Scalar(0, 255, 0), 2, 8, vector<Vec4i>(), 0, Point());
	}
	imshow("Black Display Final", blackimage1);
	imwrite("black1.png", blackimage1);
	waitKey(0);

}
