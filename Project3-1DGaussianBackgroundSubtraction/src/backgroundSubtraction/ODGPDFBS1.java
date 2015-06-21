package backgroundSubtraction;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.*;

import java.util.ArrayList;
import java.util.List;

public class ODGPDFBS1 {
	private static final int kernelSize = 5;
	private static final double stdDev = 1.5;

	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		double rho = 0.022;
		double threshold = 8.5;
		Mat currentFrame = new Mat();
		Mat grayFrame = new Mat();
		Mat blurFrame = new Mat();
		Mat smoothFrame = new Mat();

		currentFrame = Highgui.imread("/Users/adithiathreya/Desktop/SCU/COEN296-VideoProcessing/office/input/in000001.jpg");
		Imgproc.cvtColor(currentFrame, grayFrame , Imgproc.COLOR_RGB2GRAY);
		Imgproc.GaussianBlur(grayFrame, blurFrame, new Size(kernelSize,kernelSize), stdDev);

		//mean and variance
		Mat mu_t  = new Mat(blurFrame.rows(), blurFrame.cols(), CvType.CV_64FC1);
		Mat var_t = new Mat(blurFrame.rows(), blurFrame.cols(), CvType.CV_64FC1);
		Mat stdDev_t = new Mat(blurFrame.rows(), blurFrame.cols(), CvType.CV_64FC1);

		//background subtraction
		for(int r=0; r<blurFrame.rows(); ++r) {
			for(int c=0; c<blurFrame.cols(); ++c) {

				//initializing mean and variance
				double mean = 0, meanI2 = 0, mean2I = 0;
				int startM, startN, endM, endN;
				startM = (r<1?0:(r-1));
				startN = (c<1?0:(c-1));
				endM   = (r>blurFrame.rows()-2?blurFrame.rows()-1:(r+1));
				endN   = (c>blurFrame.cols()-2?blurFrame.cols()-1:(c+1));
				for (int m=startM; m<=endM; ++m) {
					for(int n=startN; n<=endN; ++n) {
						double[] pixel = blurFrame.get(r, c);
						meanI2 = meanI2 + Math.pow(pixel[0], 2);
						mean2I = mean2I + pixel[0];
					}
				}
				mean = meanI2/(endM-startM+endN-startN) - Math.pow(mean2I/(endM-startM+endN-startN), 2);
				mu_t.put(r, c, blurFrame.get(r, c)[0]);
				var_t.put(r, c, mean);
			}
		}

		for(int t=2; t<=2050; ++t) {
			
			System.out.println(t);
			//choosing N frames
			String path = String.format("/Users/adithiathreya/Desktop/SCU/COEN296-VideoProcessing/office/input/in%06d.jpg", t);
			currentFrame = Highgui.imread(path);
			Imgproc.cvtColor(currentFrame, blurFrame, Imgproc.COLOR_RGB2GRAY);
			for(int r=0; r<blurFrame.rows(); ++r) {
				for(int c=0; c<blurFrame.cols(); ++c) {
					//updating mean and variance
					double dataMu = rho*blurFrame.get(r, c)[0] + (1-rho)*mu_t.get(r, c)[0];
					mu_t.put(r, c, dataMu);
					double d = Math.abs(blurFrame.get(r, c)[0] - mu_t.get(r, c)[0]);
					double dataVar = rho*d + (1-rho)*var_t.get(r, c)[0];
					var_t.put(r, c, dataVar);
					stdDev_t.put(r, c, Math.sqrt(var_t.get(r, c)[0]));

					//classification of background/foreground
					if (d/(stdDev_t.get(r, c)[0]) > threshold) {
						blurFrame.put(r, c, 255);
					}
					else {
						blurFrame.put(r, c, 0);
					}
				}
			}
			Imgproc.GaussianBlur(blurFrame, smoothFrame, new Size(kernelSize,kernelSize), stdDev);
			path = String.format("/Users/adithiathreya/Desktop/SCU/COEN296-VideoProcessing/office/Results/in%06d.jpg", t);
			Highgui.imwrite(path, smoothFrame );
		}
	}
}
