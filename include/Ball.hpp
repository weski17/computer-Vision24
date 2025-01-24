#pragma once
#ifndef BALL_HPP
#define BALL_HPP

#include <opencv2/opencv.hpp>
#include <list>
#include <random>

// Klasse Ball, die die Eigenschaften eines Balls definiert
class Ball {
public:
    int radius, x, y;         // Koordinaten des Balls
    float vx, vy;             // Geschwindigkeit in x- und y-Richtung
    cv::Scalar color;         // Farbe des Balls

    // Konstruktor für die Initialisierung eines Balls
    Ball(int row, int col);

    // Verschiebt den Ball in die angegebene Richtung
    void move(float deltaTime, int cols);

    // Getter-Funktionen
    int getX() const { return x; }
    int getY() const { return y; }
    cv::Scalar getColor() const { return color; }
};

// Globale Liste, um alle Bälle zu speichern
extern std::list<Ball> ballList;

#endif // BALL_HPP
