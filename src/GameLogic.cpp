// GameLogic.cpp
#include "GameLogic.hpp"
#include "Video.hpp"
#include <iostream>
#include <cmath>
#include <SFML/Graphics.hpp>

GameLogic::GameLogic(TrackingPipeline &trackingPipeline)
    : tracking(trackingPipeline), score(0) {}

GameLogic::~GameLogic() {}

void GameLogic::runSingleMode(cv::VideoCapture &cap)
{
    // Create a resizable OpenCV window
    cv::namedWindow("Single Mode", cv::WINDOW_NORMAL);

    cv::Mat frame;
    sf::Clock clock;                      // Timer für das Spiel
    sf::Time timeLimit = sf::seconds(60); // Zeitbegrenzung auf 1 Minute

    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            std::cout << "Video-Tracking abgeschlossen." << std::endl;
            break;
        }

        try
        {
            // Verarbeite das Frame mit der Tracking-Pipeline
            tracking.processFrame(frame);
            const std::vector<cv::Point> &contour = tracking.getTrackedContour();

            // Füge Bälle hinzu und zeichne sie
            frame = addBallsToPicture(frame);

            // Zeichne die Kontur der verfolgten Person
            if (!contour.empty())
            {
                std::vector<std::vector<cv::Point>> contoursToDraw = {contour};
                cv::drawContours(frame, contoursToDraw, -1, cv::Scalar(0, 255, 0), 2); // Grün
            }

            // Kollisionen prüfen
            handleCollisions(contour);

            // Punktzahl anzeigen
            drawScore(frame);

            // Zeit anzeigen
            sf::Time elapsedTime = clock.getElapsedTime();
            int remainingTime = static_cast<int>(timeLimit.asSeconds() - elapsedTime.asSeconds());
            cv::putText(frame, "Time: " + std::to_string(remainingTime) + " s", cv::Point(10, 100),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

            // Wenn die Zeit abgelaufen ist, beende das Spiel
            if (elapsedTime >= timeLimit)
            {
                std::cout << "Time's up!" << std::endl;
                break;
            }

            // Ergebnis anzeigen
            cv::imshow("Single Mode", frame);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Fehler beim Tracking: " << e.what() << std::endl;
            break;
        }

        // Beende bei 'q'
        if (cv::waitKey(30) == 'q')
        {
            std::cout << "Spiel manuell beendet." << std::endl;
            break;
        }
    }

    // Endbildschirm anzeigen
    displayEndScreen();
    cv::destroyAllWindows(); // Fenster schließen
}

void GameLogic::handleCollisions(const std::vector<cv::Point> &contour)
{
    ballList.remove_if([&contour, this](Ball &ball)
                       {
        for (const cv::Point& point : contour) {
            if (cv::norm(cv::Point(ball.getX(), ball.getY()) - point) <= ball.radius) {
                // Punktelogik basierend auf der Farbe des Balls
                if (ball.getColor() == cv::Scalar(0, 255, 0)) {
                    score++; // Grüner Ball: +1 Punkt
                } else if (ball.getColor() == cv::Scalar(255, 187, 255)) {
                    score = std::max(0, score - 1); // Pinkfarbener Ball: -1 Punkt, nicht unter 0
                }
                std::cout << "Aktuelle Punktzahl: " << score << std::endl;
                return true;
            }
        }
        return false; });
}

void GameLogic::drawScore(cv::Mat &frame)
{
    cv::putText(frame, "Score: " + std::to_string(score), cv::Point(10, 50),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(238, 26, 118), 2);
    cv::putText(frame, "Winning Score: 15", cv::Point(10, frame.rows - 20),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
}

void GameLogic::displayEndScreen()
{
    sf::RenderWindow endWindow(sf::VideoMode(800, 600), "Game Over");

    sf::Texture backgroundTexture;
    if (!backgroundTexture.loadFromFile("assets/fotos/results.jpg"))
    {
        std::cerr << "Fehler beim Laden des Hintergrundbildes!" << std::endl;
        return;
    }
    sf::Sprite backgroundSprite;
    backgroundSprite.setTexture(backgroundTexture);
    backgroundSprite.setScale(
        static_cast<float>(endWindow.getSize().x) / backgroundTexture.getSize().x,
        static_cast<float>(endWindow.getSize().y) / backgroundTexture.getSize().y);

    sf::Font font;
    if (!font.loadFromFile("font/Game-Of-Squids.ttf"))
    {
        std::cerr << "Fehler beim Laden der Schriftart!" << std::endl;
        return;
    }

    sf::Text resultText;
    resultText.setFont(font);
    resultText.setCharacterSize(50);
    resultText.setFillColor(sf::Color::White);
    resultText.setStyle(sf::Text::Bold);
    resultText.setPosition(
        (endWindow.getSize().x - resultText.getLocalBounds().width) / 2,
        (endWindow.getSize().y - resultText.getLocalBounds().height) / 2);

    const int targetScore = 15; // Beispielziel
    if (score >= targetScore)
    {
        resultText.setString("Success! Score: " + std::to_string(score));
        resultText.setPosition(
            (endWindow.getSize().x - resultText.getLocalBounds().width) / 2,
            endWindow.getSize().y / 3);
        resultText.setFillColor(sf::Color::Blue);
    }
    else
    {
        resultText.setString("Failed! Score: " + std::to_string(score));
        resultText.setPosition(
            (endWindow.getSize().x - resultText.getLocalBounds().width) / 2,
            endWindow.getSize().y / 3);
        resultText.setFillColor(sf::Color::Red);
    }

    sf::Texture iconTexture;
    if (!iconTexture.loadFromFile("assets/fotos/Symbol.png"))
    {
        std::cerr << "Fehler beim Laden des Symbols!" << std::endl;
        return;
    }
    sf::Sprite iconSprite;
    iconSprite.setTexture(iconTexture);
    iconSprite.setScale(0.3f, 0.3f);

    const std::vector<std::string> options = {"New Game", "Exit"};
    std::vector<sf::Text> menuItems;
    for (size_t i = 0; i < options.size(); ++i)
    {
        sf::Text menuItem;
        menuItem.setFont(font);
        menuItem.setString(options[i]);
        menuItem.setCharacterSize(30);
        menuItem.setFillColor(i == 0 ? sf::Color(238, 26, 118) : sf::Color::White);
        menuItem.setPosition(
            (endWindow.getSize().x - menuItem.getLocalBounds().width) / 2,
            endWindow.getSize().y / 2 + i * 50);
        menuItems.push_back(menuItem);
    }

    size_t selectedItemIndex = 0;
    iconSprite.setPosition(
        menuItems[0].getPosition().x - iconSprite.getGlobalBounds().width - 10,
        menuItems[0].getPosition().y + menuItems[0].getGlobalBounds().height / 2 - iconSprite.getGlobalBounds().height / 2);

    while (endWindow.isOpen())
    {
        sf::Event event;
        while (endWindow.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
            {
                endWindow.close();
            }
            if (event.type == sf::Event::KeyPressed)
            {
                if (event.key.code == sf::Keyboard::Up)
                {
                    menuItems[selectedItemIndex].setFillColor(sf::Color::White);
                    selectedItemIndex = (selectedItemIndex == 0) ? menuItems.size() - 1 : selectedItemIndex - 1;
                    menuItems[selectedItemIndex].setFillColor(sf::Color(238, 26, 118));
                    iconSprite.setPosition(
                        menuItems[selectedItemIndex].getPosition().x - iconSprite.getGlobalBounds().width - 10,
                        menuItems[selectedItemIndex].getPosition().y + menuItems[selectedItemIndex].getGlobalBounds().height / 2 - iconSprite.getGlobalBounds().height / 2);
                }
                else if (event.key.code == sf::Keyboard::Down)
                {
                    menuItems[selectedItemIndex].setFillColor(sf::Color::White);
                    selectedItemIndex = (selectedItemIndex + 1) % menuItems.size();
                    menuItems[selectedItemIndex].setFillColor(sf::Color(238, 26, 118));
                    iconSprite.setPosition(
                        menuItems[selectedItemIndex].getPosition().x - iconSprite.getGlobalBounds().width - 10,
                        menuItems[selectedItemIndex].getPosition().y + menuItems[selectedItemIndex].getGlobalBounds().height / 2 - iconSprite.getGlobalBounds().height / 2);
                }
                else if (event.key.code == sf::Keyboard::Return)
                {
                    if (selectedItemIndex == 0)
                    {
                        endWindow.close();
                        cv::VideoCapture newCap(0);
                        if (!newCap.isOpened())
                        {
                            std::cerr << "Error: Could not open the camera." << std::endl;
                            return;
                        }
                        runSingleMode(newCap);
                    }
                    else if (selectedItemIndex == 1)
                    {
                        endWindow.close();
                    }
                }
            }
        }

        endWindow.clear();
        endWindow.draw(backgroundSprite);
        endWindow.draw(resultText);
        for (const auto &menuItem : menuItems)
        {
            endWindow.draw(menuItem);
        }
        endWindow.draw(iconSprite);
        endWindow.display();
    }
}