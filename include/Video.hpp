#pragma once
#ifndef VIDEO_HPP
#define VIDEO_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include "Ball.hpp"
#include "BallFunktions.hpp"

// Funktionen
cv::Mat addBallsToPicture(cv::Mat frame); // Fügt Bälle zum Bild hinzu

#endif // VIDEO_HPP
