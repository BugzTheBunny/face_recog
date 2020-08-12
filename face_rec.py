import face_recognition
import os
import cv2

KNOWN_DIR = 'known'
UNKNOWN_DIR = 'unknown'
# GROUP_DIR = 'groups'
MODEL = 'cnn'  # hog - Less accurate, but faster , cnn - Slow, but more accurate
TOLERANCE = 0.6  # For testing the tolerance - default 0.6

unknown_face = f'{UNKNOWN_DIR}/image.jpg'


def main():
    known_faces = []
    known_faces_encoded = []

    for image in os.listdir(KNOWN_DIR):
        image_load = face_recognition.load_image_file(f'{KNOWN_DIR}/{image}')
        image_load = cv2.cvtColor(image_load, cv2.COLOR_RGB2BGR)
        image_encoding = face_recognition.face_encodings(image_load)[0]
        known_faces.append(image_encoding)
        known_faces_encoded.append({'name': image.replace('.jpg', ''), 'encoded': image_encoding})

    try:
        unknown_face_image = face_recognition.load_image_file(unknown_face)
        unknown_face_encoding = face_recognition.face_encodings(unknown_face_image)[0]
        results = face_recognition.compare_faces(known_faces, unknown_face_encoding, tolerance=TOLERANCE)
        if True in results:
            print(f'Found {known_faces_encoded[results.index(True)]["name"]}\nTF = {results}')
        else:
            print('Did not match..')
    except IndexError:
        print("No Face Found... who dis?")
        quit()


if __name__ == '__main__':
    main()
