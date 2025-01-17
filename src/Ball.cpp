#include "Ball.hpp"

// Globale Liste für Bälle
std::list<Ball> ballList;

// Konstruktor: Erstellt einen neuen Ball mit festgelegter Farbe
Ball::Ball(int row, int col) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distX(-0.5f, 0.5f);
    std::uniform_real_distribution<float> distY(1.0f, 3.0f); // Vertikale Geschwindigkeit angepasst

    radius = std::uniform_int_distribution<int>(20, 40)(gen); // Radius erhöht
    x = std::uniform_int_distribution<int>(radius, col - radius)(gen); // Verhindere Start außerhalb des Bereichs
    y = 0; // Start oben am Bildschirmrand
    vx = distX(gen); // Geschwindigkeit in x-Richtung
    vy = distY(gen); // Geschwindigkeit in y-Richtung (sollte positiv sein)

    // Feste Farbe für alle Bälle
    color = cv::Scalar(255, 187, 255); // Pinkfarbene Bälle
}

// Verschiebt den Ball basierend auf seiner Geschwindigkeit
void Ball::move(float deltaTime, int cols) {
    x += static_cast<int>(vx * deltaTime);
    y += static_cast<int>(vy * deltaTime);

    // Begrenzung auf den sichtbaren Bereich
    if (x - radius < 0) { // Linker Rand
        x = radius;
        vx = -vx; // Richtung umkehren
    } else if (x + radius > cols) { // Rechter Rand
        x = cols - radius;
        vx = -vx; // Richtung umkehren
    }
}
