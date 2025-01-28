/**
 * @file main.cpp
 * @brief Hauptprogramm für das Eye Track-Projekt.
 *
 * Dieses Programm implementiert ein Startmenü mit SFML, führt Hintergrundsubtraktion, Multi-Object-Tracking
 * und andere Bildverarbeitungsaufgaben durch. Es verwendet OpenCV für die Bildverarbeitung
 * und SFML für die Benutzeroberfläche.
 */

#include <SFML/Graphics.hpp>
#include "startmenu.hpp"
#include "BackgroundSubtractionPipeline.hpp"
#include "TrackingPipeline.hpp"
#include "Person.hpp"
#include "MultiTracking.hpp"
#include "GameLogic.hpp"
#include <opencv2/opencv.hpp>

/**
 * @brief Der Einstiegspunkt des Programms.
 *
 * Dieses Programm zeigt ein Startmenü und führt die Bildverarbeitung auf der Kamera durch.
 * Es verwendet SFML für die Benutzeroberfläche und OpenCV für die Bildverarbeitungsfunktionen.
 *
 * @return `0`, wenn das Programm erfolgreich beendet wird. Andernfalls `-1`, wenn die
 *         Initialisierung fehlschlägt.
 */
int main()
{
    const int windowWidth = 800;  ///< Breite des SFML-Fensters.
    const int windowHeight = 600; ///< Höhe des SFML-Fensters.

    // Erstellen des SFML-Fensters
    sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "Eye Track");

    // Initialisierung der Komponenten
    StartMenu menu(windowWidth, windowHeight); ///< Startmenü-Objekt.
    BackgroundSubtractionPipeline pipeline;    ///< Pipeline für die Hintergrundsubtraktion.
    TrackingPipeline tracking;                 ///< Pipeline für das Tracking.
    MultiTracking multiTracking;               ///< Multi-Object-Tracking-Objekt.

    cv::VideoCapture cap(0); ///< Kamera als Videoquelle.
    cv::Mat groundTruthMask; ///< Ground-Truth-Maske für die Verarbeitung.

    // Überprüfen, ob die Kamera geöffnet werden kann
    if (!cap.isOpened())
    {
        std::cerr << "Fehler beim Öffnen der Kamera! Bitte überprüfen Sie die Verbindung." << std::endl;
        return -1; ///< Beenden, falls die Kamera nicht geöffnet werden kann.
    }

    // Hauptschleife für das SFML-Fenster
    while (window.isOpen())
    {
        // Ereignisverarbeitung
        menu.processEvents(window, pipeline, tracking, multiTracking, cap, groundTruthMask);

        // Fenster aktualisieren
        window.clear();
        menu.draw(window);
        window.display();
    }

    // Ressourcen freigeben
    cap.release();                  ///< Freigeben des Videoaufnahmeobjekts.
    pipeline.releaseVideoWriters(); ///< Freigeben der Videoausgabeobjekte.
    cv::destroyAllWindows();        ///< Schließen aller OpenCV-Fenster.

    return 0;
}
