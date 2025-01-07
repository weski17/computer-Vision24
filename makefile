# Compiler
CXX = g++

# Compiler-Flags
CXXFLAGS = -Wall -std=c++17 `pkg-config --cflags opencv4`

# SFML- und OpenCV-Bibliotheken
LDFLAGS = -lsfml-graphics -lsfml-window -lsfml-system -lsfml-audio `pkg-config --libs opencv4` -lstdc++fs

# Name der ausführbaren Datei
TARGET = startmenu

# Verzeichnisse für Quell- und Header-Dateien
SRC_DIR = src
INC_DIR = include

# Quell-Dateien
SRCS = $(SRC_DIR)/main.cpp \
       $(SRC_DIR)/startmenu.cpp \
       $(SRC_DIR)/BackgroundSubtractionPipeline.cpp \
       $(SRC_DIR)/TrackingPipeline.cpp \
       $(SRC_DIR)/Person.cpp \
       $(SRC_DIR)/MultiTracking.cpp

# Objekt-Dateien (erzeugt entsprechende Objektdateien im Quellverzeichnis)
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(SRC_DIR)/%.o)

# Standard-Ziel zum Erstellen des Projekts
all: $(TARGET)

# Verknüpfe Objektdateien, um die endgültige ausführbare Datei zu erstellen
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

# Kompiliere jede Quelldatei zu einer Objektdatei
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) -c $< -o $@

# Bereinige das Projekt (entferne Objektdateien und die ausführbare Datei)
clean:
	rm -f $(OBJS) $(TARGET)

# Phony-Ziele
.PHONY: all clean
