#include <opencv2/opencv.hpp>
#include "Video.hpp"

int main() {
    cv::VideoCapture cap(0); // Kamera öffnen
    if (!cap.isOpened()) {
        std::cerr << "Kamera konnte nicht geöffnet werden." << std::endl;
        return -1;
    }

    // Setze die Auflösung
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    // Erstelle das Fenster
    cv::namedWindow("Live Video mit Bällen", cv::WINDOW_NORMAL);
    cv::resizeWindow("Live Video mit Bällen", 1280, 720);

    cv::Mat frame;
    while (true) {
        cap >> frame; // Nächstes Bild von der Kamera
        if (frame.empty()) {
            std::cerr << "Kein Bild empfangen!" << std::endl;
            break;
        }

        // Bälle hinzufügen und anzeigen
        frame = addBallsToPicture(frame);
        cv::imshow("Live Video mit Bällen", frame);

        // Beenden bei Tastendruck 'q'
        if (cv::waitKey(30) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
