#include <SFML/Graphics.hpp>
#include "startmenu.hpp"
#include "BackgroundSubtractionPipeline.hpp"
#include "TrackingPipeline.hpp"
#include "Person.hpp" 
#include "MultiTracking.hpp" 
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    const int windowWidth = 800;
    const int windowHeight = 600;
    sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "Eye Track");

    StartMenu menu(windowWidth, windowHeight);
    BackgroundSubtractionPipeline pipeline;
    TrackingPipeline tracking;
    MultiTracking multiTracking;
    const std::string videoPath = "data/input/video/normaleBewegung_D.mp4";
    const std::string outputPath = "data/output/fotos";
    cv::VideoCapture cap;
    cv::Mat groundTruthMask;

if (!pipeline.initializeVideoAndLoadGroundTruth(videoPath, cap, groundTruthMask, outputPath)) {
    return -1;  // Beenden, falls Initialisierung fehlschlägt
}

    // Hauptfensterschleife
    while (window.isOpen()) {
        menu.processEvents(window, pipeline,tracking, multiTracking, cap, groundTruthMask);
        
        window.clear();
        menu.draw(window);
        window.draw(menu.getIconSprite());
        window.display();
    }

    // Ressourcen freigeben
    cap.release();
    pipeline.releaseVideoWriters();
    cv::destroyAllWindows();

    return 0;
}
