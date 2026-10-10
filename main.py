import cv2 as cv
from camera import Camera


def main():

    orbbec = None
    print("Starting Orbbec camera...")

    try:
        orbbec = Camera()
        orbbec.start()

        print("Orbbec camera live stream. Press 'q' to exit.")
        while True:
            result = orbbec.get_frames()
            if result is None:
                continue
            color_frame, depth_frame = result

            cv.imshow("Color", color_frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if orbbec is not None:
            orbbec.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
