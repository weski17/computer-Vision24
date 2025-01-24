#pragma once
#ifndef BALLFUNKTIONS_HPP
#define BALLFUNKTIONS_HPP

#include "Ball.hpp"

// Fügt neue Bälle hinzu
void addBall(int cols, int rows);

// Entfernt Bälle, die den Rahmen verlassen haben
void removeFromList(int rows, int cols);

#endif // BALLFUNKTIONS_HPP
