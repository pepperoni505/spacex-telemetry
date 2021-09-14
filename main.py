import cv2
import pafy
import math
import imutils
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR

# Resolution of image that the characters were taken from
WIDTH = 1920
HEIGHT = 1080

# Launch clock
LAUNCH_CLOCK_X_COORDS = (816, 1100) # X, X + W
LAUNCH_CLOCK_Y_COORDS = (974, 1031) # Y, Y + H

LAUNCH_CLOCK_RATIO_X = [x / WIDTH for x in LAUNCH_CLOCK_X_COORDS]
LAUNCH_CLOCK_RATIO_Y = [y / HEIGHT for y in LAUNCH_CLOCK_Y_COORDS]

# Stage 1 speed
STAGE1_SPEED_X_COORDS = (106, 226) # X, X + W
STAGE1_SPEED_Y_COORDS = (960, 1008) # Y, Y + H

STAGE1_SPEED_RATIO_X = [x / WIDTH for x in STAGE1_SPEED_X_COORDS]
STAGE1_SPEED_RATIO_Y = [y / HEIGHT for y in STAGE1_SPEED_Y_COORDS]

# Mission name
MISSION_NAME_X_COORDS = (716, 1200) # X, X + W
MISSION_NAME_Y_COORDS = (1026, 1051) # Y, Y + H

MISSION_NAME_RATIO_X = [x / WIDTH for x in MISSION_NAME_X_COORDS]
MISSION_NAME_RATIO_Y = [y / HEIGHT for y in MISSION_NAME_Y_COORDS]

MAX_SPEED_DIFFERENCE = 250 # KM/H

ocr = PaddleOCR(lang='en', use_angle_cls=True, show_log=False)

# TODO: WORK ON SOMEHOW MAKING VIDEO PLAY IN REALTIME, BETTER GRAPHING, LOG ALTITUDE AS WELL AS STAGE 2, CLEAN UP CODE

class Extract:

    def __init__(self, url):
        self.url = self._get_video_url(url, "1280x720")
        self.stream = cv2.VideoCapture(self.url)
        self.mission_name = self.get_mission_name()

    def _get_video_url(self, youtube_url, resolution):
        """
        Returns a URL to the actual video stream from any given YouTube URL

        :param youtube_url: Link to a YouTube video

        :return video_url: Link to the video stream
        """
        video = pafy.new(youtube_url)
        streams = video.allstreams
        
        # Create a dictionary of all the mp4 videos found with their resolution as the key and their url as the value
        stream_urls = dict([(s.resolution, s.url) for s in streams if (s.extension == "mp4") and (s.mediatype == "video")])

        # We default to 1080p, and go to 720p if 1080p isn't available. For now if neither are available, we throw an error. In the future, this could be improved
        if resolution in stream_urls:
            return stream_urls[resolution]
        elif "1920x1080" in stream_urls:
            return stream_urls["1920x1080"]
        elif "1280x720" in stream_urls:
            return stream_urls["1280x720"]
        else:
            raise RuntimeError("No video streams are available")

    def _clock_time_to_seconds(self, timestamp):
        h, m, s = timestamp.split(':')
        return (int(h) * 3600) + (int(m) * 60) + int(s)

    def _calculate_bounding_box(self, x_tuple, y_tuple):
        stream_height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        stream_width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        x1 = int(x_tuple[0] * stream_width)
        x2 = int(x_tuple[1] * stream_width)
        y1 = int(y_tuple[0] * stream_height)
        y2 = int(y_tuple[1] * stream_height)
        return x1, x2, y1, y2

    def _get_text_from_image(self, image):
        if image.shape[0] < 50: # PaddleOCR doesn't seem to work properly for images with a height less than 50, so we should resize the image
            image = imutils.resize(image, height=50)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, image = cv2.threshold(image, 125, 255, cv2.THRESH_BINARY)

        result = ocr.ocr(image)
        found_text = []
        for line in result:
            found_text.append(line[1][0])

        return found_text


    def find_liftoff_time(self):
        stream_fps = self.stream.get(cv2.CAP_PROP_FPS)

        self.stream.set(cv2.CAP_PROP_POS_FRAMES, (300 * stream_fps)) # SpaceX streams tend to start around the 5 minute mark, so start our search there.

        x1, x2, y1, y2 = self._calculate_bounding_box(LAUNCH_CLOCK_RATIO_X, LAUNCH_CLOCK_RATIO_Y)

        while True:
            ret, frame = self.stream.read()
            if ret:
                current_frame = self.stream.get(cv2.CAP_PROP_POS_FRAMES)
                result = self._get_text_from_image(frame[y1: y2, x1:x2])
                for text in result:
                    if text.startswith('T-') and len(text) == 10:
                        # Remove "T-" from text
                        text = text[2:]
                        # Get seconds from the text
                        total_seconds = self._clock_time_to_seconds(text)

                        # Convert seconds to frame
                        frame_number = total_seconds * stream_fps
                        return frame_number + current_frame

                self.stream.set(cv2.CAP_PROP_POS_FRAMES, current_frame + (30 * stream_fps))

    def get_mission_name(self):
        x1, x2, y1, y2 = self._calculate_bounding_box(MISSION_NAME_RATIO_X, MISSION_NAME_RATIO_Y)
        liftoff_frame = self.find_liftoff_time()
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, liftoff_frame)

        ret, frame = self.stream.read()
        if ret:
            result = self._get_text_from_image(frame[y1:y2, x1:x2])
            for text in result:
                if len(text) > 5: # Arbitrary length to check for
                    return text

    def get_time_since_liftoff(self):
        return

    def get_telemetry(self):
        # Find liftoff
        liftoff_frame = self.find_liftoff_time() # TODO: Start at T-10 and detect when T- switches to T+
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, liftoff_frame)

        stream_fps = self.stream.get(cv2.CAP_PROP_FPS)

        x1, x2, y1, y2 = self._calculate_bounding_box(STAGE1_SPEED_RATIO_X, STAGE1_SPEED_RATIO_Y)

        plt.xlabel('Seconds')
        plt.ylabel('KM/H')
        # plt.xlim([0, 500])
        # plt.ylim([0, 10000])
        plt.ion()
        plt.show()
        x_data = []
        y_data = []
        while True:
            ret, frame = self.stream.read()
            if ret:
                current_frame = self.stream.get(cv2.CAP_PROP_POS_FRAMES) # multi threading processing
                if current_frame % stream_fps == 0.0:
                    result = self._get_text_from_image(frame[y1:y2, x1:x2])
                    for text in result:
                        if text.isdigit():
                            previous = 0
                            if y_data:
                                previous = y_data[-1]
                            if math.fabs(int(text) - previous) < MAX_SPEED_DIFFERENCE:
                                x_data.append(int((current_frame - liftoff_frame) / stream_fps))
                                y_data.append(int(text))
                                plt.clf()
                                plt.plot(x_data, y_data)
                    plt.draw()
                    plt.pause(0.001)

                cv2.imshow('video', frame)
                cv2.waitKey(int(1000 / stream_fps))


def main():
    extract = Extract("https://www.youtube.com/watch?v=QJXxVtp3KqI")
    extract.get_telemetry()


if __name__ == '__main__':
    main()