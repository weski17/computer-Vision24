#include <opencv2/opencv.hpp>
#include <iostream>
#include "Video.hpp"

int main() {
    std::cout << "Starte das Programm..." << std::endl;

    // Kamera öffnen
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Fehler: Kamera konnte nicht geöffnet werden!" << std::endl;
        return -1;
    }
    std::cout << "Kamera erfolgreich geöffnet." << std::endl;

    // Fenster erstellen
    cv::namedWindow("Live Video mit Bällen", cv::WINDOW_AUTOSIZE);

    cv::Mat frame;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Fehler: Frame konnte nicht abgerufen werden!" << std::endl;
            break;
        }

        frame = addBallsToPicture(frame);

        cv::imshow("Live Video mit Bällen", frame);

        char key = cv::waitKey(30);
        if (key == 'q') {
            std::cout << "Taste 'q' gedrückt, beende das Programm." << std::endl;
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    std::cout << "Programm beendet." << std::endl;

    return 0;
}
