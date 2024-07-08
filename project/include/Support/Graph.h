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
        int rows;
        int columns;
        int old_id;
        int old_op_id;
    };
    Graph::Node& add_node(int id, std::string data, int old_id, int old_op_id);
    Graph::Node& add_node_with_dims(int id, std::string data, int rows, int columns, int old_id, int old_op_id);
    std::map<int, Node> get_nodes() {
        return nodes;
    }
private:
    bool directed;
    std::map<int, Node> nodes;
};

#endif
