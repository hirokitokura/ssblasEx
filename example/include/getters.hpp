#ifndef SSBLAS_EXAMPLE_GETTERS
#define SSBLAS_EXAMPLE_GETTERS

ssblasIntType_t getINTTYPE(std::string str)
{
    ssblasIntType_t INT_type;
    //std::cout << "getINTTYPE"<<str << std::endl;
    if(str.compare("int")==0){
        //printf("getINTTYPE: int\n");
        INT_type=SSBLAS_int_32;
    }
    else if(str.compare("longint")==0){
        //printf("getINTTYPE: longint\n");
        INT_type=SSBLAS_int_64;
    }
    else{
        std::cout << "UNKNOWN INT TYPE "<<str << std::endl;
        exit(-1);
    }
    return INT_type;
}
ssblasDataType_t getSCALETYPE(std::string str)
{
    //std::cout << "getSCALETYPE: " << str << std::endl;
    ssblasDataType_t SCALE_type;
    if(str.compare("float")==0){
        //std::cout << "getSCALETYPE: " << "float" << std::endl;
        SCALE_type=SSBLAS_R_32F;
    }
    else if(str.compare("double")==0){
        //std::cout << "getSCALETYPE: " << "double" << std::endl;
        SCALE_type=SSBLAS_R_64F;
    }
    else if(str.compare("int")==0){
        //std::cout << "getSCALETYPE: " << "int" << std::endl;
        SCALE_type=SSBLAS_R_32I;
    }
    else if(str.compare("char")==0){
        //std::cout << "getSCALETYPE: " << "char" << std::endl;
        SCALE_type=SSBLAS_R_8I;
    }
    else{
        std::cout << "UNKNOWN SCALE TYPE "<<str << std::endl;
        exit(-1);
    }
    return SCALE_type;
}
ssblasDataType_t getABTYPE(std::string str)
{
    //std::cout << "getABTYPE: " << str << std::endl;
    ssblasDataType_t SCALE_type;
    if(str.compare("float")==0){
        SCALE_type=SSBLAS_R_32F;
    }
    else if(str.compare("double")==0){
        SCALE_type=SSBLAS_R_64F;
    }
    else if(str.compare("char")==0){
        SCALE_type=SSBLAS_R_8I;
    }
    else{
        std::cout << "UNKNOWN AB TYPE "<<str << std::endl;
        exit(-1);
    }
    return SCALE_type;
}
ssblasDataType_t getCTYPE(std::string str)
{
    ssblasDataType_t SCALE_type;
    if(str.compare("float")==0){
        SCALE_type=SSBLAS_R_32F;
    }
    else if(str.compare("double")==0){
        SCALE_type=SSBLAS_R_64F;
    }
    else if(str.compare("int")==0){
        SCALE_type=SSBLAS_R_32I;
    }
    else{
        std::cout << "UNKNOWN C TYPE "<<str << std::endl;
        exit(-1);
    }
    return SCALE_type;
}


ssblasOperation_t getTRANS(std::string str)
{
    ssblasOperation_t TRANS_type;
    if(str.compare("N")==0){
        TRANS_type=SSBLAS_OP_N;
    }
    else if(str.compare("T")==0){
        TRANS_type=SSBLAS_OP_T;
    }
    else if(str.compare("C")==0){
        TRANS_type=SSBLAS_OP_C;
    }
    else{
        std::cout << "UNKNOWN TRANS TYPE "<<str << std::endl;
        exit(-1);
    }
    return TRANS_type;
}

template <typename DEF_DOUBLE_AB>
ssblasDataType_t getDataType()
{
    if constexpr (std::is_same_v<DEF_DOUBLE_AB, float>) {
        return SSBLAS_R_32F;
    }
    else if constexpr (std::is_same_v<DEF_DOUBLE_AB, double>) {
        return SSBLAS_R_64F;
    }
    else if constexpr (std::is_same_v<DEF_DOUBLE_AB, int>) {
        return SSBLAS_R_32I;
    }
    else if constexpr (std::is_same_v<DEF_DOUBLE_AB, char>) {
        return SSBLAS_R_8I;
    }
    else{
        printf("getDataType\n");
        exit(-1);
    }
}

double getTimeinS()
{
    //printf("getTimeinS\n");
    double retval;
    retval=omp_get_wtime();
    //printf("getTimeinS\n");
    return retval; 
}



template <typename INDEXINT>
std::string getNameINDEXINT()
{
    if constexpr (std::is_same_v<INDEXINT, int >) {
        return "int";
    }
    else if constexpr (std::is_same_v<INDEXINT, long int>) {
        return "longint";
    }
    else{
        exit(-1);
    }
}

template <typename DEF_DOUBLE>
std::string getNameDEF_DOUBLE()
{
    if constexpr (std::is_same_v<DEF_DOUBLE, float>) {
        return "float";
    }
    else if constexpr (std::is_same_v<DEF_DOUBLE, double>) {
        return "double";
    }
    else if constexpr (std::is_same_v<DEF_DOUBLE, int>) {
        return "int";
    }
    else if constexpr (std::is_same_v<DEF_DOUBLE, char>) {
        return "char";
    }
    else{
        exit(-1);
    }
}


std::string getTRANS(ssblasOperation_t TRANS_type)
{
   if(SSBLAS_OP_N==TRANS_type){
        return "N";
   }
   else if(SSBLAS_OP_T==TRANS_type){
        return "T";
   }
   else if(SSBLAS_OP_C==TRANS_type){
        return "C";
   }
}
#endif