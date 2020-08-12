import face_recognition
import os

IMAGE_2_TEST = 'image'

# Folders
KNOWN_DIR = 'known'
UNKNOWN_DIR = 'unknown'
MODEL = 'cnn'  # 'hog' - Less accurate, but much faster , 'cnn' - Slower but more accurate.
TOLERANCE = 0.6  # For testing the tolerance - default 0.6 (higher -> less accurate)

unknown_face = f'{UNKNOWN_DIR}/{IMAGE_2_TEST}.jpg'


def main():
    # "database"
    known_faces = []
    known_faces_encoded = []

    for image in os.listdir(KNOWN_DIR):
        # loads the image.
        image_load = face_recognition.load_image_file(f'{KNOWN_DIR}/{image}')
        # encodes the image to the face_recognition format - array for each face.
        image_encoding = face_recognition.face_encodings(image_load)[0]  # Selecting the first array
        # adding the encoding version into an array.
        known_faces.append(image_encoding)
        # adding into a "temp db"
        known_faces_encoded.append({'name': image.replace('.jpg', ''), 'encoded': image_encoding})

    try:
        # Loading the image that should be compared - in a try catch, the library expects to find a face.
        unknown_face_image = face_recognition.load_image_file(unknown_face)
        unknown_face_encoding = face_recognition.face_encodings(unknown_face_image)[0]
        # So this line does all the magic, it tries to compare each face from the db to the unknown face
        results = face_recognition.compare_faces(known_faces, unknown_face_encoding, tolerance=TOLERANCE)
        if True in results:
            # Just a pretty print with some data.
            print(f'Found {known_faces_encoded[results.index(True)]["name"]}\nTF = {results}')
        else:
            print('Did not match..')
    except IndexError:
        print("No Face Found... who dis?")
        quit()


if __name__ == '__main__':
    main()
