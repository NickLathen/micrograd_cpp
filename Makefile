CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Wpedantic

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

SOURCES = $(SRC_DIR)/micrograd.cpp $(SRC_DIR)/test.cpp
OBJECTS = $(OBJ_DIR)/micrograd.o $(OBJ_DIR)/test.o
EXECUTABLE = $(BIN_DIR)/test

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(EXECUTABLE)

.PHONY: all clean
