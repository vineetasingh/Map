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
const int areathresh = 200;//75
const int areath = 500;
const int noofruns = 35;//35
const int disti = 5;
const int hood = 3; // checking for edge points in a (hood X hood) area
const float stdgaus = 25;//19.8843;
const float meangaus = 96.8514;// 96.8514;
const int psudoB = 0;
const int psudoG = 128;
const int psudoR = 255;
RNG rng(12345);
void coverup(Mat  & blackimage)
{
	Rect cover = Rect(750, 0, 250, 224); // coordinates ( 807,0) (807,224), (1023, 0), (1023, 224)
	Mat roi_cover = blackimage(cover);
	roi_cover = cv::Scalar(104, 40, 14);
	imwrite("tempm.png", blackimage);

}


Mat findedges(Mat blackimage)
{
	Mat newimg = blackimage;
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
	
	Mat cm_img0;
	addWeighted(abs_grad_x, 2, abs_grad_y, 2, 0, grad);
	applyColorMap(grad, cm_img0, COLORMAP_HOT);

	cv::inRange(cm_img0, cv::Scalar(250,250,250), cv::Scalar(255, 255, 255), grad);
	imshow("edges", grad); imwrite("edges.png", grad);
	
	
	// Show the result:
	imshow("cm_img0", cm_img0);
	imwrite("colormap.png", cm_img0);
	//removebackground(cm_img0);

	

	return grad;
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


vector<vector<Point>> movesomemore(vector<vector<Point>> DrawCont, Mat blackimage)
{
	vector<Point2f> centro = findcentroid(DrawCont); int xpo, ypo; vector<Point> cont; vector<vector<Point>> contsto;
	for (int id = 0; id < DrawCont.size(); id++)
	{
		for (int pd = 0; pd < DrawCont[id].size(); pd++)
		{
			double A = angletime(DrawCont[id][pd].x, DrawCont[id][pd].y, centro[id].x + 15, centro[id].y, centro[id].x, centro[id].y);// finding angle at each point of the contour

			gotonewpoint(DrawCont[id][pd].x, DrawCont[id][pd].y, xpo, ypo, A, 15);
			cont.push_back(Point(xpo, ypo));
		}
		contsto.push_back(cont);
		cont.clear();
	}
	return contsto;
}


int isitanedgepoints(Mat edges, int x, int y)
{
	int edgpoints = 0;
	if ((x>10) && (x<1010) && (y>10) && (y < 1010))
	{
		for (int i = x - hood; i <= x + hood; i++)
		{
			for (int j = y - hood; j <= y + hood; j++)
			{
				if (edges.at<uchar>(j, i) == 0) //checking presense of edge points around a neigbourhood
					edgpoints++;
			}
		}
	}
	if (edgpoints > 8)
		return 1;
	else
		return 0;
}

vector<vector<Point>> outercontours(Mat grad)
{

	// --------------------Getting the outer contours based on edges-------------------------------------
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	vector<vector<Point>> mixed;
	dilate(grad, grad, Mat());
	dilate(grad, grad, Mat());
	dilate(grad, grad, Mat());
	dilate(grad, grad, Mat());
	dilate(grad, grad, Mat());
	dilate(grad, grad, Mat());
	dilate(grad, grad, Mat());
	findContours(grad, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); // Find the contours in the image


	return contours;
}

vector<vector<Point>> mixcontours(Mat grad, Mat newimg, vector<vector<Point>> blackcont, vector<vector<Point>> outercont)
{
	Mat blackimage = imread("16.5/16.5.4/2015-012-003_LMM.24.4.30_C57Bl6_E16.5_NKX2.1_HOPX_ACTA2-01-01.png");
	//Mat blackimage = imread("16.5/16.5.15/2015-001-013_LMM.14.24.4.40_C57Bl6_E16.5_HOPX-pHISH3_ACTA2-08.png");
	vector<vector<Point>> blackcontt; vector<vector<Point>> outercontt; vector<float> asprat(outercont.size());
	vector<float> ipts; vector<float> opts; vector<float>dipts; vector<float> dopts;


	vector<Rect> outerrem(outercont.size());
	for (int i = 0; i < outercont.size(); i++)
	{
		outerrem[i] = boundingRect(Mat(outercont[i]));
		asprat[i] = outerrem[i].width / outerrem[i].height;
	}


	// Save eonly the biggest contours
	for (int i = 0; i < outercont.size(); i++)
	{
		float chaido = fabs(contourArea(cv::Mat(outercont[i])));
		if ((chaido > areathresh) && (asprat[i] <= 1))
		{
			outercontt.push_back(outercont[i]); //saving only the large contours in another variable
		}
	}

	for (int j = 0; j < blackcont.size(); j++)
	{
		float chaidb = fabs(contourArea(cv::Mat(blackcont[j])));
		if (chaidb > areathresh)
		{
			blackcontt.push_back(blackcont[j]); //saving only the large contours in another variable
		}
	}


	// Draw bounding rectangle around each of the contours
	vector<Rect> outerrect(outercontt.size());
	vector<Rect> blackrect(blackcontt.size());
	int count = 0;
	for (int i = 0; i < outercontt.size(); i++)
		outerrect[i] = boundingRect(Mat(outercontt[i]));

	for (int j = 0; j < blackcontt.size(); j++)
		blackrect[j] = boundingRect(Mat(blackcontt[j]));

	Mat creatimgo(grad.size(),CV_8U);
	creatimgo.setTo(cv::Scalar(0,0,0)); 

	Mat creatimgi(grad.size(), CV_8U);
	creatimgi.setTo(cv::Scalar(0, 0, 0));

	Mat dst(grad.size(), CV_8U);
	dst.setTo(cv::Scalar(0, 0, 0));
	
	for (int i = 0; i < outerrect.size(); i++)
	{
		int count = 0;
		int xo1 = outerrect[i].x; int xo2 = xo1 + outerrect[i].width;
		int yo1 = outerrect[i].y; int yo2 = (outerrect[i].y) + outerrect[i].height;
		
		for (int j = 0; j < blackrect.size(); j++)
		{
			int countc = 0;
			int xb1 = blackrect[j].x; int xb2 = (blackrect[j].x) + blackrect[j].width;
			int yb1 = blackrect[j].y; int yb2 = (blackrect[j].y) + blackrect[j].height;
			if (((xb1> xo1) && (xb1<xo2)) && ((yb1> yo1) && (yb1 < yo2)))
				countc++;
			if (((xb2> xo1) && (xb2<xo2)) && ((yb1> yo1) && (yb1 < yo2)))
				countc++;
			if (((xb1> xo1) && (xb1<xo2)) && ((yb2> yo1) && (yb2 < yo2)))
				countc++;
			if (((xb2> xo1) && (xb2<xo2)) && ((yb2> yo1) && (yb2 < yo2)))
				countc++;
			

			if (countc >= 2)
			{
				++count;

				if ((count >= 3)) // keep the inner contours based on their count inside the bigger contours
					ipts.push_back(j);
				
			}

		}
		if (count == 0)   // if there are no black contour inside the outer contour
			opts.push_back(i);
		if (count == 1)  // if there is one black contour inside the outer contour
			opts.push_back(i);
		if (count == 2)  // if there are two black contour inside the outer contour
			opts.push_back(i);
		//if (count >= 3) // if there are greater than or equal to 3 black contour inside the outer contour
			//cout << count << endl;
	}

	/*
	//-----------------checking the number of outer contours inside blackcont  ------------------------
	for (int i = 0; i < blackrect.size(); i++)
	{
		int boount = 0;
		int xo1 = blackrect[i].x; int xo2 = xo1 + blackrect[i].width;
		int yo1 = blackrect[i].y; int yo2 = (blackrect[i].y) + blackrect[i].height;

		for (int j = 0; j < outerrect.size(); j++)
		{
			int countb = 0;
			int xb1 = outerrect[j].x; int xb2 = (outerrect[j].x) + outerrect[j].width;
			int yb1 = outerrect[j].y; int yb2 = (outerrect[j].y) + outerrect[j].height;
			if (((xb1> xo1) && (xb1<xo2)) && ((yb1> yo1) && (yb1 < yo2)))
				countb++;
			if (((xb2> xo1) && (xb2<xo2)) && ((yb1> yo1) && (yb1 < yo2)))
				countb++;
			if (((xb1> xo1) && (xb1<xo2)) && ((yb2> yo1) && (yb2 < yo2)))
				countb++;
			if (((xb2> xo1) && (xb2<xo2)) && ((yb2> yo1) && (yb2 < yo2)))
				countb++;


			if (countb >= 3)
			{
				++boount;
				dopts.push_back(j);
			}
		}


		if (boount >= 1)   // if there are no black contour inside the outer contour
			ipts.push_back(i);


		//if (count >= 3) // if there are greater than or equal to 3 black contour inside the outer contour
		//cout << count << endl;
	}
*/

		//rectangle(creatimgo, outerrect[i].tl(), outerrect[i].br(), Scalar(255, 255, 255), CV_FILLED, 8, 0);
		
	
		/*imshow("creat  image outer", creatimgo);
		imshow("creat  image inner", creatimgi);
		imshow("and  image outer", dst);*/
	

	/*for (int i = 0; i < outerrect.size(); i++)
	{
		count = 0;
		for (int j = 0; j < blackrect.size(); j++)
		{
			bool is_inside = ((blackrect[j] & outerrect[i]) == blackrect[j]);//(bool is_inside = (outerrect[i] & blackrect[j]) == blackrect[j])
			if (is_inside == 1)
				count++;
			
		}
	}*/
	// drawing the final contours after applying some conditions
	for (int i = 0; i < blackcontt.size(); i++)
	{
		for (int y = 0; y < ipts.size(); y++)
		{
			if (i == ipts[y])
			{
				Scalar color1 =Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
				cout << "saved" << endl;
				drawContours(blackimage, blackcontt, i, Scalar(0, 125, 255), 2, 8, vector<Vec4i>(), 0, Point());
			}
		}
		
	}
	for (int i = 0; i < outercontt.size(); i++)
	{
		for (int y = 0; y < opts.size(); y++)
		{
			//for (int h = 0; h < dopts.size(); h++)
			//{
				Scalar color2 = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
				if ((i == opts[y]))
					drawContours(blackimage, outercontt, i, Scalar ( 0,255,0), 2, 8, vector<Vec4i>(), 0, Point());
			
		}
	}

	imshow("final set", blackimage);
	imwrite("final_set.png", blackimage);
	return outercont;
}

// rectangle inside each other using coordinates code trial

   
//===================================================MAIN STARTS==========================================================

int main()
{
	// =============================== Initialization===============================================

	Mat blackimage = imread("16.5/16.5.4/2015-012-003_LMM.24.4.30_C57Bl6_E16.5_NKX2.1_HOPX_ACTA2-01-01.png"); //Load source image
	//Mat blackimage = imread("16.5/16.5.15/2015-001-013_LMM.14.24.4.40_C57Bl6_E16.5_HOPX-pHISH3_ACTA2-08.png");
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

	/*~~~~~CALL~~~~~~*/ Mat edges = findedges(blackimage2);
	/*~~~~~CALL~~~~~~*/findgradient(blackimage1, grad_x, grad_y, grad);// finds gradient of the whole image once in the program


	// ##  STEP 1 : Find good black contours  Contours
	//===== ======================= Thresholding &  Contouring===============================================
	Mat thr_blackimage(blackimage.rows, blackimage.cols, CV_8UC1);
	GaussianBlur(blackimage, blackimage, Size(7, 7), 0, 0, 4);  // removing unnecessary noisy background
	cv::inRange(blackimage, cv::Scalar(0, 0, 0), cv::Scalar(35, 45, 45), thr_blackimage);//35, 30, 30)  35, 45, 35
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
											// ##  END STEP 1


	/*~~~~~CALL~~~~~~*/vector<Point2f> centroi = findcentroid(contours);// finding centroid of each relaevant contour once for the image
	
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ DRAW CONTOURS, IMSHOW & END ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	vector<vector<Point>> tempacontours(contours.size()); vector<vector<Point>> mcontours(tempacontours.size()); vector<vector<Point>> anotcontours(contours.size());


	for (int i = 0; i < contours.size(); i++) // going through each initial contour
	{
		cout << "Inside loop %d" << i+1<< endl;
		cun = 0;//count of number of edges
		int countofedge = 0;

		//FIRST RUN : Moving the points in the contour by 5 and checking if edge point
		// moving points ahead and storing it in tempacontours
		for (int j = 0; j < contours[i].size(); j++)// going through points in every contour
		{
			cout << "Inside 1"<< endl;
			pin = contours[i][j];// a particular point j in contour i
			A = angletime(pin.x, pin.y, centroi[i].x + 15, centroi[i].y, centroi[i].x, centroi[i].y);// finding angle at each point of the contour
			gotonewpoint(pin.x, pin.y, xpo, ypo, A, diout);
			int uu = isitanedgepoints(edges, xpo, ypo);
			if (uu == 1)//yes, it is a edge point
			{
				mcontours[i].push_back(Point(xpo, ypo));
				countofedge++;
			}

			else
				tempacontours[i].push_back(Point(xpo, ypo));
		}
		int ycon = 0;
	DOP: if ((countofedge < (0.8*contours[i].size()))&& (ycon<200))
	
	{
		ycon++;
		cout << "Inside 2" << endl;
		anotcontours[i].clear();
		for (int q = 0; q < tempacontours[i].size(); q++)
			anotcontours[i].push_back(Point(tempacontours[i][q].x, tempacontours[i][q].x));

		tempacontours[i].clear();
		for (int qi = 0; qi < anotcontours[i].size(); qi++)
		{
			int xpoo = 0, ypoo = 0;
			Point yin = anotcontours[i][qi];
			A = angletime(yin.x, yin.y, centroi[i].x + 15, centroi[i].y, centroi[i].x, centroi[i].y);// finding angle at each point of the contour
			gotonewpoint(yin.x, yin.y, xpoo, ypoo, A, diout);
			if (isitanedgepoints(edges, xpoo, ypoo) == 1)
			{
				mcontours[i].push_back(Point(xpoo, ypoo));
				countofedge++;
			}
			else
			{
				tempacontours[i].push_back(Point(xpoo, ypoo)); 
				goto DOP;
			}
		}
	}
		 


		}


		// SUBSEQUENT RUNS : Moving the points in the contour by 5 and checking if edge point till atleast half the points in the contour are edge points





		/*if (cun < tempacontours[i].size() / 2) // if (no of edge points in the contour < half the points in the contour)--> go for another run
		{
			cun = 0;
		pop:
			for (int jo = 0; jo < tempacontours[i].size(); jo++)// going through points in every contour
			{
				pin = tempacontours[i][jo];// a particular point jo in contour i
				//pin = mcontours[i][jo];
				A = angletime(pin.x, pin.y, centroi[i].x + 15, centroi[i].y, centroi[i].x, centroi[i].y);// finding angle at each point of the contour
				gotonewpoint(pin.x, pin.y, xpo, ypo, A, diout);
				mcontours[i].push_back(Point(xpo, ypo)); // new point now put into mcontours
				if (isitanedgepoints(edges, xpo, ypo) == 1)
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
	*/



	/*~~~~~CALL~~~~~~*/vector<vector<Point>>lapo = movesomemore(mcontours, blackimage1);
	/*~~~~~CALL~~~~~~*/vector<vector<Point>> outercont = outercontours(edges);
	/*~~~~~CALL~~~~~~*/vector<vector<Point>> mixcont = mixcontours(edges, untouched, lapo, outercont);

	for (int ido = 0; ido < lapo.size(); ido++)
	{
		drawContours(blackimage1, lapo, ido, Scalar(0, 255, 0), 2, 8, vector<Vec4i>(), 0, Point());
	}
	imshow("Black Display Final", blackimage1);
	imwrite("black1.png", blackimage1);
	waitKey(0);

}
