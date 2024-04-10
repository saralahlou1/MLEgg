#ifndef GRAPH_H
#define GRAPH_H

#include <string>

class Graph {
public:
    Graph(bool directed = true);
    void add_edge();
    void to_file(const std::string& file);
private:
};

#endif
