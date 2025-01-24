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
 * Dieses Programm zeigt ein Startmenü und führt die Bildverarbeitung auf einem Video durch.
 * Es verwendet SFML für die Benutzeroberfläche und OpenCV für die Bildverarbeitungsfunktionen.
 *
 * @return `0`, wenn das Programm erfolgreich beendet wird. Andernfalls `-1`, wenn die
 *         Initialisierung fehlschlägt.
 */
int main() {
    const int windowWidth = 800; ///< Breite des SFML-Fensters.
    const int windowHeight = 600; ///< Höhe des SFML-Fensters.
    
    // Erstellen des SFML-Fensters
    sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "Eye Track");

    // Initialisierung der Komponenten
    StartMenu menu(windowWidth, windowHeight); ///< Startmenü-Objekt.
    BackgroundSubtractionPipeline pipeline; ///< Pipeline für die Hintergrundsubtraktion.
    TrackingPipeline tracking; ///< Pipeline für das Tracking.
    MultiTracking multiTracking; ///< Multi-Object-Tracking-Objekt.

    const std::string videoPath = "data/input/video/ueberlappungAus.mp4"; ///< Pfad zum Eingabevideo.
    const std::string outputPath = "data/output/fotos"; ///< Pfad zum Ausgabeverzeichnis.
    cv::VideoCapture cap; ///< OpenCV-Videoaufnahmeobjekt.
    cv::Mat groundTruthMask; ///< Ground-Truth-Maske für die Verarbeitung.

    // Initialisierung der Videoverarbeitung und Laden der Ground-Truth-Maske
    if (!pipeline.initializeVideoAndLoadGroundTruth(videoPath, cap, groundTruthMask, outputPath)) {
        return -1; ///< Beenden, falls die Initialisierung fehlschlägt.
    }

    // Hauptschleife für das SFML-Fenster
    while (window.isOpen()) {
        // Ereignisverarbeitung
        menu.processEvents(window, pipeline, tracking, multiTracking, cap, groundTruthMask);

        // Fenster aktualisieren

        window.clear();
        menu.draw(window);
        window.display();
    }

    // Ressourcen freigeben
    cap.release(); ///< Freigeben des Videoaufnahmeobjekts.
    pipeline.releaseVideoWriters(); ///< Freigeben der Videoausgabeobjekte.
    cv::destroyAllWindows(); ///< Schließen aller OpenCV-Fenster.

    return 0;
}
