//
// Author: marvin on 11/20/19.
//

#ifndef ATHOM_H
#define ATHOM_H

class Athom {
private:
    int m_posX;
    int m_posY;
public:
    Athom() : m_posX(0), m_posY(0){}
    Athom(int _m_posX, int _m_posY) : m_posX(_m_posX), m_posY(_m_posY) {}
    virtual double activationFunction(double x) = 0;
    virtual double activationFunctionDerivative(double x) = 0;
};


#endif //ATHOM_H
