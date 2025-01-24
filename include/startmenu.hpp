#ifndef STARTMENU_HPP
#define STARTMENU_HPP

#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>
#include "BackgroundSubtractionPipeline.hpp"
#include "TrackingPipeline.hpp"
#include "MultiTracking.hpp" 
#include <opencv2/opencv.hpp>
#define MAX_NUMBER_OF_ITEMS 4  ///< Anzahl der Menüoptionen

// Enum für die Menüoptionen
enum MenuOption {
    BackgroundSubtraction = 0,
    SingleMode,
    MultiMode,
    Exit
};

/**
 * @class StartMenu
 * @brief Verwaltet das grafische Hauptmenü der Anwendung, einschließlich Navigation, Anzeige und Auswahl.
 */
class StartMenu {
public:
    /**
     * @brief Konstruktor, initialisiert das Menü mit Schriftarten, Texten, Symbolen und Musik.
     * @param width Breite des Fensters
     * @param height Höhe des Fensters
     */
    StartMenu(float width, float height);

    /// Destruktor
    ~StartMenu();

    /**
     * @brief Zeichnet das Menü und die Hintergrundgrafiken im Renderfenster.
     * @param window Referenz auf das Fenster, in dem das Menü gezeichnet wird
     */
    void draw(sf::RenderWindow &window);
    
    /// Bewegt die Auswahl im Menü eine Position nach oben.
    void MoveUp();

    /// Bewegt die Auswahl im Menü eine Position nach unten.
    void MoveDown();

    /// Gibt den Index der aktuell ausgewählten Menüoption zurück.
    int GetPressedItem() const { return selectedItemIndex; }

    /// Gibt eine Referenz auf das Symbol-Sprite zurück.
    sf::Sprite& getIconSprite();
    
    /**
     * @brief Aktualisiert die Symbolposition basierend auf der ausgewählten Option.
     * @param windowWidth Breite des Fensters zur Positionsberechnung
     */
    void updateIconPosition(float windowWidth);

    /// Startet die Hintergrundmusik.
    void startMusic();

    /// Stoppt die Hintergrundmusik.
    void stopMusic();

    /**
     * @brief Verarbeitet Benutzerereignisse und leitet Menüoptionen ein.
     * @param window Referenz auf das Fenster
     * @param pipeline Referenz auf die Hintergrundsubtraktionspipeline
     * @param cap Videoquelle zur Verarbeitung
     * @param groundTruthMask Ground-Truth-Maske für Hintergrundsubtraktion
     */
    void processEvents(sf::RenderWindow &window, BackgroundSubtractionPipeline &pipeline, TrackingPipeline &tracking, MultiTracking &multiTracking,
                       cv::VideoCapture &cap, const cv::Mat &groundTruthMask);

    // Hilfsmethoden zur Positionierung der Menüoptionen und Symbole
    sf::Vector2f getPositionOfItem(int index); ///< Gibt die Position der Menüoption zurück
    float getTextWidth(int index);             ///< Gibt die Textbreite der Menüoption zurück
    float getTextHeight(int index);            ///< Gibt die Texthöhe der Menüoption zurück

private:
    /**
     * @brief Verarbeitet die Auswahl einer Menüoption und führt Aktionen entsprechend aus.
     * @param selectedOption Die gewählte Menüoption
     * @param window Referenz auf das Fenster
     * @param pipeline Referenz auf die Hintergrundsubtraktionspipeline
     * @param cap Videoquelle zur Verarbeitung
     * @param groundTruthMask Ground-Truth-Maske für Hintergrundsubtraktion
     */
    void handleMenuSelection(MenuOption selectedOption, sf::RenderWindow &window, BackgroundSubtractionPipeline &pipeline,TrackingPipeline &tracking, MultiTracking &multiTracking,
                             cv::VideoCapture &cap, const cv::Mat &groundTruthMask);

    int selectedItemIndex;                    ///< Aktuell ausgewählte Menüoption
    sf::Font font;                            ///< Schriftart für Menüoptionen
    sf::Text menu[MAX_NUMBER_OF_ITEMS];       ///< Array für Menütexte
    sf::Texture backgroundTexture;            ///< Hintergrundbildtextur
    sf::Sprite backgroundSprite;              ///< Hintergrund-Sprite
    sf::Texture iconTexture;                  ///< Symboltextur für Auswahl
    sf::Sprite iconSprite;                    ///< Symbol-Sprite
    sf::Music backgroundMusic;                ///< Hintergrundmusik
};

#endif  // STARTMENU_HPP