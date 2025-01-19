#include "startmenu.hpp"
#include "GameLogic.hpp"
#include <iostream>

/**
 * @brief Konstruktor: Initialisiert das Menü, lädt Schriftarten, Bilder und Musik, und setzt die Position der Menüoptionen.
 * 
 * @param width Breite des Fensters
 * @param height Höhe des Fensters
 */
StartMenu::StartMenu(float width, float height) {
    const float verticalOffset = 35.0f;

    // Schriftart laden
    if (!font.loadFromFile("font/Game-Of-Squids.ttf")) {
        std::cerr << "Fehler beim Laden der Schriftart!" << std::endl;
    }

    // Hintergrundbild laden und skalieren
    if (!backgroundTexture.loadFromFile("assets/fotos/hauptmenu.jpg")) {
        std::cerr << "Fehler beim Laden des Hintergrundbildes!" << std::endl;
    }
    backgroundSprite.setTexture(backgroundTexture);
    backgroundSprite.setScale(
        static_cast<float>(width) / backgroundTexture.getSize().x,
        static_cast<float>(height) / backgroundTexture.getSize().y
    );

    // Symbol laden und skalieren
    if (!iconTexture.loadFromFile("assets/fotos/Symbol.png")) {
        std::cerr << "Fehler beim Laden des Symbols!" << std::endl;
    }
    iconSprite.setTexture(iconTexture);
    iconSprite.setScale(0.3f, 0.3f);

    // Hintergrundmusik laden und starten
    if (!backgroundMusic.openFromFile("assets/audio/Voicy_Squid-Game-OST-Pink-Soldiers.ogg")) {
        std::cerr << "Fehler beim Laden der Hintergrundmusik!" << std::endl;
    }
    backgroundMusic.setLoop(true);
    backgroundMusic.play();

    // Menüoptionen konfigurieren
    const std::vector<std::string> options = {"BG Subtraction", "Single Mode", "Multi Mode", "Exit"};
    for (int i = 0; i < MAX_NUMBER_OF_ITEMS; ++i) {
        menu[i].setFont(font);
        menu[i].setString(options[i]);
        menu[i].setFillColor(i == 0 ? sf::Color(255, 0, 29) : sf::Color::White);  // Markiert als ausgewählte Option
        menu[i].setCharacterSize(25);
        menu[i].setPosition((width - menu[i].getGlobalBounds().width) / 2, 
                            height / (MAX_NUMBER_OF_ITEMS + 1) * (i + 1) + verticalOffset);
    }

    selectedItemIndex = 0;  // Standardauswahl auf die erste Option setzen
    updateIconPosition(width);  // Position des Symbols neben der ersten Menüoption setzen
}

StartMenu::~StartMenu() {}

/**
 * @brief Zeichnet das Hintergrundbild und die Menüoptionen im angegebenen Renderfenster.
 * 
 * @param window Referenz auf das Renderfenster
 */
void StartMenu::draw(sf::RenderWindow &window) {
    window.draw(backgroundSprite);
    for (int i = 0; i < MAX_NUMBER_OF_ITEMS; i++) {
        window.draw(menu[i]);
    }
}

/**
 * @brief Bewegt die Menüauswahl eine Position nach oben und aktualisiert das Symbol.
 */
void StartMenu::MoveUp() {
    if (selectedItemIndex - 1 >= 0) {
        menu[selectedItemIndex].setFillColor(sf::Color::White);  // Alte Auswahlfarbe zurücksetzen
        selectedItemIndex--;
        menu[selectedItemIndex].setFillColor(sf::Color(255, 0, 29));  // Neue Auswahl markieren
        updateIconPosition(backgroundSprite.getGlobalBounds().width);  // Symbol neu positionieren
    }
}

/**
 * @brief Bewegt die Menüauswahl eine Position nach unten und aktualisiert das Symbol.
 */
void StartMenu::MoveDown() {
    if (selectedItemIndex + 1 < MAX_NUMBER_OF_ITEMS) {
        menu[selectedItemIndex].setFillColor(sf::Color::White);  // Alte Auswahlfarbe zurücksetzen
        selectedItemIndex++;
        menu[selectedItemIndex].setFillColor(sf::Color(255, 0, 29));  // Neue Auswahl markieren
        updateIconPosition(backgroundSprite.getGlobalBounds().width);  // Symbol neu positionieren
    }
}

/**
 * @brief Aktualisiert die Position des Menü-Symbols basierend auf der aktuell ausgewählten Menüoption.
 * 
 * @param windowWidth Breite des Fensters zur Berechnung der Symbolposition.
 */
void StartMenu::updateIconPosition(float windowWidth) {
    sf::Vector2f menuItemPosition = getPositionOfItem(selectedItemIndex);
    float textWidth = getTextWidth(selectedItemIndex);
    float textHeight = getTextHeight(selectedItemIndex);
    float textMiddleY = menuItemPosition.y + textHeight / 2;
    float symbolMiddleY = textMiddleY - iconSprite.getGlobalBounds().height / 2;

    iconSprite.setPosition(
        (windowWidth - textWidth) / 2 - iconSprite.getGlobalBounds().width - 10, 
        symbolMiddleY
    );
}

/**
 * @brief Startet die Hintergrundmusik.
 */
void StartMenu::startMusic() {
    if (backgroundMusic.getStatus() != sf::Music::Playing) {
        backgroundMusic.play();
    }
}

/**
 * @brief Stoppt die Hintergrundmusik.
 */
void StartMenu::stopMusic() {
    if (backgroundMusic.getStatus() == sf::Music::Playing) {
        backgroundMusic.stop();
    }
}

/**
 * @brief Gibt die Position der Menüoption für den angegebenen Index zurück.
 */
sf::Vector2f StartMenu::getPositionOfItem(int index) {
    return index >= 0 && index < MAX_NUMBER_OF_ITEMS ? menu[index].getPosition() : sf::Vector2f(0, 0); 
}

/**
 * @brief Gibt die Breite des Textes der Menüoption zurück.
 */
float StartMenu::getTextWidth(int index) {
    return index >= 0 && index < MAX_NUMBER_OF_ITEMS ? menu[index].getGlobalBounds().width : 0;
}

/**
 * @brief Gibt die Höhe des Textes der Menüoption zurück.
 */
float StartMenu::getTextHeight(int index) {
    return index >= 0 && index < MAX_NUMBER_OF_ITEMS ? menu[index].getGlobalBounds().height : 0;
}

/**
 * @brief Gibt eine Referenz auf das Symbol-Sprite zurück.
 */
sf::Sprite& StartMenu::getIconSprite() {
    return iconSprite;
}

/**
 * @brief Verarbeitet die Auswahl der Menüoption und führt die entsprechende Aktion aus.
 * 
 * @param selectedOption Die ausgewählte Menüoption
 * @param window Referenz auf das Renderfenster
 * @param pipeline Referenz auf die Pipeline zur Hintergrundsubtraktion
 * @param cap Die Videoquelle
 * @param groundTruthMask Die Ground-Truth-Maske
 */
void StartMenu::handleMenuSelection(MenuOption selectedOption, sf::RenderWindow &window,
                                    BackgroundSubtractionPipeline &pipeline, TrackingPipeline &tracking,
                                    cv::VideoCapture &cap,
                                    const cv::Mat &groundTruthMask) {




    
                                     
    switch (selectedOption) {
        case BackgroundSubtraction: {
            std::cout << "Background Subtraction gewählt." << std::endl;
            window.close();

            cv::Mat frame;
            
            // VideoWriters in der Pipeline initialisieren
            pipeline.initializeVideoWriters("data/output/video", cap);

            while (true) {
                cap >> frame;
                if (frame.empty()) {
                    std::cout << "Video abgeschlossen, Verarbeitung beendet." << std::endl;
                    break;
                }

                try {
                    pipeline.saveBackgroundSubtractionResults(frame, groundTruthMask);
                } catch (const std::exception &e) {
                    std::cerr << "Fehler bei Background Subtraction: " << e.what() << std::endl;
                    break; // Beende Verarbeitung bei Fehler
                }

                if (cv::waitKey(30) == 'q') {
                    std::cout << "Background Subtraction manuell beendet." << std::endl;
                    break;
                }
            }
            pipeline.releaseVideoWriters();
            cv::destroyAllWindows(); // Fenster schließen
            break;
        }
        case SingleMode: {
            std::cout << "Single Mode gestartet." << std::endl;
            window.close();

            // Erstelle ein TrackingPipeline-Objekt
            TrackingPipeline tracking;

            // Erstelle die Spiel-Logik
            GameLogic gameLogic(tracking);

            // Starte den Single Mode
            gameLogic.runSingleMode(cap);
            break;
        }
        case MultiMode:
            std::cout << "Multi Mode (später hinzuzufügen)" << std::endl;
            break;
        case Exit:
            std::cout << "Programm wird beendet." << std::endl;
            window.close();
            cv::destroyAllWindows(); // Alle Fenster schließen
            break;
    }
}

/**
 * @brief Verarbeitet die Benutzerereignisse im Menü und leitet Aktionen ein.
 * 
 * @param window Referenz auf das Renderfenster
 * @param pipeline Referenz auf die Pipeline zur Hintergrundsubtraktion
 * @param cap Die Videoquelle
 * @param groundTruthMask Die Ground-Truth-Maske
 */
void StartMenu::processEvents(sf::RenderWindow &window, BackgroundSubtractionPipeline &pipeline, TrackingPipeline &tracking,
                              cv::VideoCapture &cap, const cv::Mat &groundTruthMask) {
    sf::Event event;
    while (window.pollEvent(event)) {
        if (event.type == sf::Event::Closed) {
            window.close();
        }
        if (event.type == sf::Event::KeyPressed) {
            if (event.key.code == sf::Keyboard::Up) {
                MoveUp();
            } else if (event.key.code == sf::Keyboard::Down) {
                MoveDown();
            } else if (event.key.code == sf::Keyboard::Return) {
                stopMusic();
                MenuOption selectedOption = static_cast<MenuOption>(GetPressedItem());
                handleMenuSelection(selectedOption, window, pipeline,tracking, cap, groundTruthMask);
            }
        }
    }
}