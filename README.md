# corner-annotator

How to use it?
1. Drag and drop a file, mp4 or jpg
2. You can visualize the image, combining the gray image, corners detected by Harris corner detector of OpenCV, and Hough line detector. You can turn on/off the features with checkbox and adjust weights with slider. Non-Maximum Suppression can be also applied to the final result.
3. In order to annotate a point, click the "Click" button and click the point you want to annotate. If the point is correct you can click "yes" in the message box that shows up after clicking the point.
4. By checking "Auto Correction" checkbox, You can use auto-correction function which automatically choose the maximum value of feature image, you set before with the checkboxs and sliders, around the point you click. The feature image excludes the gray image even you set to visualize it. If you don't want to use the function and click a point manually, just uncheck the "Auto Correction" checkbox.
5. If you want to insert a "Unknown" point which is required for your format, you can click the "Insert None" button.
6. A result example is like below. '-' means the "None" point you cannot.

484, 305
823, 479
-
913, 295

