#include "Video.hpp"

cv::Mat addBallsToPicture(cv::Mat frame) {
    int cols = frame.cols; // Breite des Bildes
    int rows = frame.rows; // Höhe des Bildes

    addBall(cols, rows);         // Füge neue Bälle hinzu
    removeFromList(rows, cols);  // Entferne Bälle, die aus dem Bild verschwinden

    for (Ball& ball : ballList) {
        ball.move(1.0f, cols); // Bewege den Ball, jetzt mit Breite
        if (ball.x >= 0 && ball.x < cols && ball.y >= 0 && ball.y < rows) {
            cv::circle(frame, cv::Point(ball.getX(), ball.getY()), ball.radius, ball.getColor(), -1); // Ball zeichnen
        }
    }
    return frame;
}
