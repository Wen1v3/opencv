import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

WIDTH = 640 * 1.2
HEIGHT = 360 * 1.2


class CameraApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Camera Feed with Number")

        self.number = tk.IntVar()
        self.number.set(10)

        self.number2 = tk.IntVar()
        self.number2.set(200)

        self.toggle_value = True

        self.label = tk.Label(master)
        self.label.grid(row=0, column=0)

        self.label_develop = tk.Label(master)
        self.label_develop.grid(row=0, column=1)

        self.label_staging = tk.Label(master)
        self.label_staging.grid(row=1, column=0)

        self.slider_frame = tk.Frame(master)
        self.slider_frame.grid(row=1, column=1)

        # Create a slider to set the number
        self.slider = tk.Scale(
            self.slider_frame,
            from_=0,
            to_=100,
            orient=tk.HORIZONTAL,
            label="Canny:",
            variable=self.number,
        )
        self.slider.pack(pady=5)

        # Create a slider to set the number
        self.slider2 = tk.Scale(
            self.slider_frame,
            from_=0,
            to_=300,
            orient=tk.HORIZONTAL,
            label="White Threshold:",
            variable=self.number2,
        )
        self.slider2.pack(pady=5)

        self.toggle = tk.Button(self.slider_frame, text="Fix Transform", command=self.toggle)
        self.toggle.pack(pady=5)

        self.corners = []

        # Capture the camera feed
        self.capture = cv2.VideoCapture(1)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

        # Start updating the frame
        self.update_frame()

    def toggle(self):
        self.toggle_value = not self.toggle_value

    def update_label(self, label, frame):
        # Convert the frame to RGBA
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        # Convert to ImageTk format
        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the label with the new image
        label.img_tk = img_tk
        label.config(image=img_tk)

    def transform_bord(self, image, corners):
        # order: top-left, top-right, bottom-right, bottom-left
        pts1 = np.array(corners, dtype=np.float32)

        h, w = image.shape[:2]
        width, height = h, h
        pts2 = np.array(
            [
                [0, 0],
                [width, 0],
                [width, height],
                [0, height],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        maxWidth = int(
            max(np.linalg.norm(pts2[1] - pts2[0]), np.linalg.norm(pts2[2] - pts2[1]))
        )
        maxHeight = int(
            max(np.linalg.norm(pts2[2] - pts2[3]), np.linalg.norm(pts2[1] - pts2[0]))
        )

        # warpPerspective does not modify input image
        return cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))

    def detect_stone(self, image, center, radius):
        x, y = center
        roi = image[y - radius : y + radius, x - radius : x + radius]

        if roi.size == 0:
            return 0

        mean_color = np.mean(roi)
        return mean_color

    def get_distance_point(self, contour, corner):
        farthest_distance = 0
        nearest_distance = 1000000
        farthest_point = corner
        nearest_point = corner

        for point in contour:
            distance = np.linalg.norm(point[0] - np.array(corner))

            if distance > farthest_distance:
                farthest_distance = distance
                farthest_point = point[0]
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_point = point[0]
        return nearest_point, farthest_point

    def update_frame(self):
        # Read a frame from the camera
        ret, frame = self.capture.read()
        if ret:
            # Overlay the number on the frame
            number = self.number.get()
            number2 = self.number2.get()

            # for some windows
            staging = frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # for other windows
            develop = blur.copy()

            edges = cv2.Canny(blur, number / 100 * 255, number / 100 * 255 * 3)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

            contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
            if len(contours_sorted) >= 3:
                contour = contours_sorted[2]

                # calculate attribute of selected contour
                hull = cv2.convexHull(contour)
                perimeter = cv2.arcLength(hull, True)
                area = cv2.contourArea(contour)

                # draw selected contour
                cv2.drawContours(develop, [contour], -1, (0, 255, 0), 1)
                cv2.putText(
                    develop,
                    "perimeter:" + str(int(perimeter)),
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    develop,
                    "area:" + str(int(area)),
                    (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

                # get board corners
                height, width = develop.shape[:2]
                bottom_left = self.get_distance_point(contour, (0, height))[0]
                bottom_right = self.get_distance_point(contour, (width, height))[0]
                top_left = self.get_distance_point(contour, bottom_right)[1]
                top_right = self.get_distance_point(contour, bottom_left)[1]
                cv2.circle(develop, tuple(bottom_left), 5, (0, 255, 0), -1)
                cv2.circle(develop, tuple(bottom_right), 5, (0, 255, 0), -1)
                cv2.circle(develop, tuple(top_left), 5, (0, 255, 0), -1)
                cv2.circle(develop, tuple(top_right), 5, (0, 255, 0), -1)

                # toggle
                if self.toggle_value:
                    self.corners = [top_left, top_right, bottom_right, bottom_left]

                points = np.vstack(self.corners)

                # transform
                staging = self.transform_bord(staging, points)
                staging_gray = cv2.cvtColor(staging, cv2.COLOR_BGR2GRAY)
                staging_gray = cv2.equalizeHist(staging_gray)

                height, width = staging.shape[:2]
                for i in range(11):
                    for j in range(11):
                        x = int(i * height / 12 + height / 12)
                        y = int(j * height / 12 + height / 12)

                        value = int(self.detect_stone(staging_gray, [x, y], 10))

                        if value > number2:
                            color = (255, 0, 0)
                        elif value < 20:
                            color = (0, 0, 255)
                        else:
                            color = (255, 255, 255)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(
                            staging,
                            str(value),
                            (x, y),
                            font,
                            0.3,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.rectangle(
                            staging,
                            tuple([x - 10, y - 10]),
                            tuple([x + 10, y + 10]),
                            color,
                            1,
                        )

            self.update_label(self.label, frame)
            self.update_label(self.label_develop, develop)
            self.update_label(self.label_staging, staging)

        # Schedule the next frame update after 3 seconds
        self.master.after(300, self.update_frame)

    def __del__(self):
        # Release the camera when the app is closed
        self.capture.release()


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
