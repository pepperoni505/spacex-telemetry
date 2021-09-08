import os
import cv2
import numpy as np
import pafy
from math import fabs

# Credit to https://github.com/shahar603/SpaceXtract for providing an idea on how to extract text from an image

def get_video_url(youtube_url):
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
    if "1920x1080" in stream_urls:
        return stream_urls["1920x1080"]
    elif "1280x720" in stream_urls:
        return stream_urls["1280x720"]
    else:
        raise RuntimeError("No video streams are available")

def remove_duplicates(list, min_distance=10):
    """
    Removes duplicate digits that are within a certain distance of each other

    :param list: A list of tuples containing the character, location, and probability
    :param min_distance: Minimum distance for a digit to occur
    """
    remove_list = [] # I tried this with removing directly from the list, but that didn't seem to work, so we have a seperate list to remove things from. This isn't ideal, so might want to fix in the future
    for i in list:
        for j in list:
            if i != j:
                if fabs(i[1] - j[1]) < min_distance:
                    if i[2] > j[2]: # Remove the value with the lowest probability
                        remove_list.append(j)
                    else:
                        remove_list.append(i)

    for i in remove_list:
        if i in list:
            list.remove(i)
    
    list.sort(key = lambda x: x[1])
    return list
    
def get_text_from_image(image, template_dir, characters, min_probability=0.75):
    """
    Extract text from a given image

    :param image: Image to search
    :param template_dir: Path to character templates
    :param characters: List of all of the characters in the template directory. This must be in the same order as the files
    :param min_probability: The minimum probability to find a character in an image.

    :return string: String from the characters found in the image
    """
    if not os.path.exists(template_dir):
        raise FileNotFoundError("Couldn't locate template images")

    found_characters = []
    character_num = 0
    for (_, _, filename) in os.walk(template_dir):
        for file in filename:
            template_path = os.path.join(template_dir, file)
            template_image = cv2.imread(template_path)

            digit_res = cv2.matchTemplate(template_image, image, cv2.TM_CCOEFF_NORMED)
            
            loc = np.where(digit_res >= min_probability)

            for pt in zip(*loc[::-1]):
                character = characters[character_num]
                position = pt[0]
                probability = digit_res[pt[1]][pt[0]]

                found_characters.append((character, position, probability))

            character_num += 1

    found_characters = remove_duplicates(found_characters)

    found_text = ''.join([x[0] for x in found_characters])
    
    return found_text

def find_liftoff_time(stream):
    stream_fps = stream.get(cv2.CAP_PROP_FPS)
    stream_height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stream_width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    frame_time = int(1000 / stream_fps) # Get how long each frame should appear in milliseconds
    while True:
        ret, frame = stream.read()
        if ret:
            cv2.imshow('video', frame)
            cv2.waitKey(frame_time)


def main():
    current_directory = os.path.dirname(os.path.realpath(__file__))
    characters_path = os.path.join(os.path.dirname(current_directory), 'characters/telemetry') # came from 1080p
    test_image = cv2.imread(os.path.join(current_directory, "test.png"))
    characters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-"]
    text = get_text_from_image(test_image, characters_path, characters, 0.75)
    print(text)
    # stream_url = get_video_url("https://www.youtube.com/watch?v=QJXxVtp3KqI")
    # stream = cv2.VideoCapture(stream_url)
    # find_liftoff_time(stream)
    



if __name__ == '__main__':
    main()