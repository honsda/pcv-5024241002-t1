import os

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE, "crazy.jpeg")
VIDEO_PATH = os.path.join(BASE, "drift.mp4")

CHANNELS = [("original", "Original"), ("red", "Red"),
            ("green", "Green"), ("blue", "Blue")]


def keep_channel(img, channel):
    out = np.zeros_like(img)
    if channel == "original":
        out[:] = img
    elif channel == "blue":
        out[:, :, 0] = img[:, :, 0]
    elif channel == "green":
        out[:, :, 1] = img[:, :, 1]
    elif channel == "red":
        out[:, :, 2] = img[:, :, 2]
    return out


def to_display(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def resize_width(img, width):
    out = to_display(img)
    h, w = out.shape[:2]
    return cv2.resize(out, (width, int(width * h / w)),
                      interpolation=cv2.INTER_AREA)


def save_result(filename, img, width):
    small = resize_width(img, width)
    cv2.imwrite(os.path.join(BASE, filename), small)
    return small


def show_result(title, filename, img, width=480):
    small = save_result(filename, img, width)
    cv2.imshow(title, small)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_image():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {IMAGE_PATH}")

    for channel, title in CHANNELS:
        show_result(title, f"hasil-image-{channel}.png",
                    keep_channel(img, channel), 480)


def show_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video tidak ditemukan: {VIDEO_PATH}")

    first_frame = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for channel, title in CHANNELS:
            small = resize_width(keep_channel(frame, channel), 240)
            cv2.imshow(title, small)

            if first_frame:
                cv2.imwrite(os.path.join(BASE, f"hasil-video-{channel}.png"),
                            small)

        first_frame = False

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    show_image()
    show_video()


if __name__ == "__main__":
    main()
