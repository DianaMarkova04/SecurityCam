import mediapipe as mp
print("Files in directory:", __file__)
print(dir(mp))
try:
    print(mp.solutions)
    print("Found solutions")
except AttributeError:
    print("No solutions attribute")
