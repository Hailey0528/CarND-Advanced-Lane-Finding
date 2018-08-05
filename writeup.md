## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**
The aim of this project is detecting lane boundary, determining the curvature of the lane and the vehicle position with respect to center. camera calibration is done at first. Color transform and gradients are used to create a thresholded binary image. Then the a perspective transform is deployed to retify binary image. From the binary image the lane pixels are detected and a second order polynomial is obtained from the detected lane pixels. After that the lane boundary has been warped back onto the original image.

[//]: # (Image References)

[image1]: ./output_images/undistortion_chessboard.jpg "undistort_chessboard"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


### Camera Calibration

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. The code for this step is contained in lines 12 through 37 of the file called `pipeline.py`).  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients, which is implemented in line 42 of the file called 'pipeline.py'  I applied this distortion correction to one calibration image, which is implemented in line 47 of the file called 'pipeline.py'. I obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Distortion Correction

After applying the distortion correction from camera calibration to one test image, the result is this one:
![alt text][image2]

#### 2. Threshlding

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 57 through 81 in function called `threshold()` in `pipeline.py`).  Here's an example of my output for this step.  

![alt text][image3]

#### 3. Perspective Transform

The code for my perspective transform includes a function called `unwarper()`, which appears in lines 83 through 93 in the file `pipeline.py`. The `unwarper()` function takes as inputs an binary image (`img_binary`), as well as source (`src`) and destination (`dst`) points. 

The source and destination points, which I have chosen are:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 450      | 100, 0        | 
| 180, 700      | 100, 700      |
| 1200, 700     | 1200, 700      |
| 700, 450      | 1200, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Finding Lane Lane
Then I created the class which names in find_lane_pixels to identify lane-line pixels.The code for this step is contained in in lines 97 through 177 of the file called `pipeline.py`. 

Because the lane lines are in the bottom half ot the image, I take a histogram of the bottom half of the image in the vertical direction. Then the peaks of the left and right halves of the histogram are identified. After that, I chose the number of the sliding windows, the width of the windows, the height of the windows.

After thees preparations, in each window, the window boundaries in y direction, in x direction for left and right lane, will be calculated. Then check the nonzero pixels in x and y direction within the windows. The indices of the pixels should be saved. If the number of the pixels is above the the chosen minimum number of pixels, then recenter next window to their mean position. After I found all the pixels which fulfilled the all the requirements, I obtained the pixel positions of left and right lane line.

Then I created one class, which names fit_polynomial to fit a second order polynomial to fit the two lane lines. The code for this step is contained in lines 217 through 235 of the file called 'pipeline.py'. Using np.polyfit a second order polynomial is obtained with the pixel positions from last step. In this class, I also generated all the y values, which means in the whole y direction. Then the values in x direction are also generated based on the fitted second order polynomial and the generated y values.

![alt text][image5]

#### Radius of Curvature and Offset of Vehicle

Next the radius of the curvature of the lane is calculated, which is in the class measure_curvature. The code for this step is contained in lines 237 through 246 of the file called 'pipeline.py'. With all the x and y positions in meter a second polynomial is obtained. The parameter of transformation from pixel position to real position in meter are xm_per_pix = 3.7/700, ym_per_pix = 30/720, which are from Udacity Course. Then I need to calculate the current curvature, which means when the y position is  the maximal value within all the y postions.

Curvature is calculated with:

![first equation](https://latex.codecogs.com/gif.latex?R_%7Bcurve%7D%20%3D%20%5Cfrac%7B%281&plus;%28%5Cfrac%7B%5Cpartial%20x%20%7D%7B%5Cpartial%20y%7D%29%5E%7B2%7D%29%29%5E%7B%5Cfrac%7B3%7D%7B2%7D%7D%7D%7B%5Cleft%20%7C%20%5Cfrac%7B%5Cpartial%5E2%20x%7D%7B%5Cpartial%20y%5E2%7D%5Cright%20%7C%7D)

Then I calculated the curvature with the average of the left and right curvature.

#### Plot Back Down

I implemented this step in lines 275 through 292 in my code in `pipeline.py` in the function process_image.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
