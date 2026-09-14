import cv2


def denoise_for_detection(frame):
    """Cleans up a disturbed frame before handing it to the detectors — accuracy
    matters more here than visual fidelity, so this runs ONLY on the copy fed
    to the detectors, never on what's actually displayed on screen."""
    denoised = cv2.fastNlMeansDenoisingColored(
        frame, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
    )
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
