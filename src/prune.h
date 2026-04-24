#ifndef MLIP_PRUNE_H
#define MLIP_PRUNE_H

#include "mtpr.h"
#include <string>

// Inherit from MLMTPR to access protected basis members
class Prune : public MLMTPR
{
public:
    Prune(const std::string &config_path);
    ~Prune() = default;
    void run();

private:
    std::string config_path_;
};

#endif