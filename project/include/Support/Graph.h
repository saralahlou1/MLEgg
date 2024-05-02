#ifndef GRAPH_H
#define GRAPH_H

#include <map>
#include <string>
#include <vector>

//template <typename T>
class Graph {
public:
    Graph(bool directed = true) : directed(directed) {}
    void to_file(const std::string& filename);
    Graph from_file(const std::string& filename);

    struct Node {
        std::string data;
        std::vector<int> children;
    };
    Graph::Node& add_node(int id, std::string data);
    std::map<int, Node> get_nodes() {
        return nodes;
    }
private:
    bool directed;
    std::map<int, Node> nodes;
};

#endif
