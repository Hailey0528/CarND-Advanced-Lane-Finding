#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
from numpy.linalg import inv

def plot(img_original, img_transform, title_original, title_transform, img_name):
	# Plot the result
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
	f.tight_layout()
	ax1.imshow(img_original)
	ax1.set_title(title_original, fontsize=40)
	ax2.imshow(img_transform, cmap='gray')
	ax2.set_title(title_transform, fontsize=40)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	f.savefig('output_images/'+img_name)

### pipeline for image processing
# Read in and make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Array to store object points and image points from all the images
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image

# Prepare object points and image points of all chessboard images
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) #x, y coordinates

for fname in images:
	# Read in each image
	img = mpimg.imread(fname)

	# Convert image to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	# Find the chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

	# If corners are found, add object points, image points
	if ret == True:
		imgpoints.append(corners)
		objpoints.append(objp)
	
## Based on image points and object points perform camera calibration and image distortion correction to the chessboard image

# Perform the camera calibration to the chessboard image
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape,None,None)	

# Apply image distortion correction to chessboard image
undist = cv2.undistort(img, mtx, dist, None, mtx)

# Save the original and the undistorted image of this image
plot(img, undist, 'Original Image', 'Undistorted Image', 'undistortion_chessboard.jpg')


## Based on image points and object points perform camera calibration and image distortion correction to the test image

# reading in an test image
img_test = mpimg.imread('test_images/test2.jpg')

# Perform the camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape,None,None)	

# Apply image distortion correction to test image
undist_test = cv2.undistort(img_test, mtx, dist, None, mtx)

# Save the original and the undistorted image of this image
plot(img_test, undist_test, 'Original Image', 'Undistorted Image', 'undistortion_testImage.jpg')

def threshold(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(sxbinary == 1) | (s_binary == 1)] = 1

    return combined_binary

img_binary = threshold(undist_test)

plot(img_test, img_binary, 'Original Image', 'Binary Image', 'image_binary.jpg')

print(img.shape[0], img.shape[1])

def unwarp(img_binary):
	# Four source points 
	src = np.float32([[580, 450], [180, 700], [1200, 700], [700, 450]])
	# Four destination points
	img_size = (img.shape[1], img.shape[0])
	dst = np.float32([[100, 0], [100, 700], [1200, 700], [1200, 0]])
	# Given src and dst points, calculate the perspective transform matrix
	M = cv2.getPerspectiveTransform(src, dst)
	# Warp the image using OpenCV warpPerspective()
	warped = cv2.warpPerspective(img_binary, M, img_size, flags=cv2.INTER_LINEAR)
	return warped, M

warped, M = unwarp(img_binary)
plot(img_binary, warped, 'Original Image', 'Warped Image', 'warped.jpg')

src = np.array([[580, 450], [180, 700], [1200, 700], [700, 450]], np.int32)
src = src.reshape((-1, 1, 2))
test=cv2.polylines(img_test, [src], True, (0, 0, 255))
plt.imshow(test)
cv2.imwrite('output_images/test.jpg', test)

# Take a histogram of the bottom half of the image
histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(histogram)
fig.savefig('output_images/test2.jpg')


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox <= win_xleft_high)
        & (nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)).nonzero()[0]
        good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox <= win_xright_high)
        & (nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if (len(good_left_inds) >= minpix):
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if (len(good_left_inds) >= minpix):
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        #pass # Remove this when you add your function

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped, ym_per_pix, xm_per_pix):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to fit the twp lane lines
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    out_img[pd.astype(int),left_fitx.astype(int)] = [255, 255, 0]
    out_img[ploty.astype(int), right_fitx.astype(int)] = [255, 255, 0]

    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    return out_img, ploty, left_fit_cr, right_fit_cr, left_fitx, right_fitx

ym_per_pix = 30/720 
xm_per_pix = 3.7/700
out_img, ploty, left_fit_cr, right_fit_cr, left_fitx, right_fitx = fit_polynomial(warped, ym_per_pix, xm_per_pix)

plot(img_test, out_img, 'Original', 'Lane Pixels', 'Finding_Lane_Pixels.jpg')

def measure_curvature(ploty, left_fit, right_fit, ym_per_pix, xm_per_pix):
    # Define y-value where we want radius of curvature
    y_eval = np.max(ploty)
    # Implementation of the calculation of R-curve
    left_curverad = ((1+(2*left_fit[0]*y_eval*ym_per_pix+left_fit[1])**2)**1.5)/np.absolute(2*left_fit[0])
    right_curverad = ((1+(2*right_fit[0]*y_eval*ym_per_pix+right_fit[1])**2)**1.5)/np.absolute(2*right_fit[0])
  
    return left_curverad, right_curverad

left_curverad, right_curverad = measure_curvature(ploty, left_fit_cr, right_fit_cr, ym_per_pix, xm_per_pix)

curverad = (left_curverad + right_curverad)/2
    
print(left_curverad, right_curverad)
offset = ((right_fitx[-1]+left_fitx[-1])/2-img.shape[1]/2)*xm_per_pix
print(offset)

# Create an image to draw the lines on
warp_zero = np.zeros_like(warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, inv(M), (img_test.shape[1], img_test.shape[0])) 
# Combine the result with the original image
result = cv2.addWeighted(undist_test, 1, newwarp, 0.3, 0)


font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(result, "curvature radius: {} m".format(str(round(curverad, 3))), (50,50), font, 1, (255, 255, 255),2,cv2.LINE_AA)
cv2.putText(result, "offset from center: {} m".format(str(round(offset, 3))), (50,100), font, 1, (255, 255, 255), 2,cv2.LINE_AA)
plt.imshow(result)
cv2.imwrite('output_images/result.jpg', result)
left_fit = np.array([])
left_fit = np.array([1, 2, 3])
left_fit = np.empty(shape=[3,])
print(left_fit is None)
print(left_fit.shape)
