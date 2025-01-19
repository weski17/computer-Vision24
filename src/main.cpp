#include <SFML/Graphics.hpp>
#include "startmenu.hpp"
#include "BackgroundSubtractionPipeline.hpp"
#include "TrackingPipeline.hpp"
#include "GameLogic.hpp"
#include <opencv2/opencv.hpp>

int main()
{
    const int windowWidth = 800;
    const int windowHeight = 600;

    // Erstelle ein Fenster für das Menü
    sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "Start Menu");

    // Initialisiere das Startmenü
    StartMenu startMenu(windowWidth, windowHeight);

    // Initialisiere Tracking-Pipeline und andere benötigte Komponenten
    TrackingPipeline tracking;
    BackgroundSubtractionPipeline pipeline;
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Fehler: Webcam konnte nicht geöffnet werden!" << std::endl;
        return -1;
    }
    cv::Mat groundTruthMask; // Leere Maske (für spätere Nutzung)

    // Menü-Schleife
    while (window.isOpen())
    {
        // Verarbeite Benutzerereignisse im Menü
        startMenu.processEvents(window, pipeline, tracking, cap, groundTruthMask);

        // Zeichne das Menü
        window.clear();
        startMenu.draw(window);
        window.display();
    }

    return 0;
}
