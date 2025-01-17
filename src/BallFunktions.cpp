#include "BallFunktions.hpp"

// Fügt neue Bälle hinzu, falls Platz ist
void addBall(int cols, int rows) {
    if (ballList.size() < 10) { // Maximal 10 Bälle
        ballList.emplace_back(rows, cols);
    }
}

// Entfernt Bälle, die den Rahmen verlassen haben
void removeFromList(int rows, int cols) {
    ballList.remove_if([rows, cols](const Ball& ball) {
        return ball.y > rows + ball.radius; // Entferne Bälle, die aus dem Bildschirm fallen
    });
}
