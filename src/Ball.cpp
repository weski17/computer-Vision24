// Anpassungen in Ball.cpp
#include "Ball.hpp"
#include <random>

// Globale Liste für Bälle
std::list<Ball> ballList;

// Konstruktor: Erstellt einen neuen Ball mit zufälliger Farbe
Ball::Ball(int row, int col)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distX(-0.5f, 0.5f);
    std::uniform_real_distribution<float> distY(1.0f, 3.0f);

    radius = std::uniform_int_distribution<int>(20, 40)(gen);
    x = std::uniform_int_distribution<int>(radius, col - radius)(gen);
    y = 0; // Start oben am Bildschirmrand
    vx = distX(gen);
    vy = distY(gen);

    // Zufällige Farbzuweisung
    color = (std::uniform_int_distribution<int>(0, 1)(gen) == 0)
                ? cv::Scalar(0, 255, 0)      // Grün
                : cv::Scalar(255, 187, 255); // Pink
}

// Verschiebt den Ball basierend auf seiner Geschwindigkeit
void Ball::move(float deltaTime, int cols)
{
    x += static_cast<int>(vx * deltaTime);
    y += static_cast<int>(vy * deltaTime);

    if (x - radius < 0)
    {
        x = radius;
        vx = -vx;
    }
    else if (x + radius > cols)
    {
        x = cols - radius;
        vx = -vx;
    }
}
