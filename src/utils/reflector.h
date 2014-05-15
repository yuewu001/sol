/*************************************************************************
	> File Name: reflector.h
	> Copyright (C) 2014 Yue Wu<yuewu@outlook.com>
	> Created Time: 2014/5/12 Monday 16:21:00
	> Functions: C++ reflector
	************************************************************************/
#ifndef HEADER_CPP_REFLECTOR
#define HEADER_CPP_REFLECTOR

#include <map>
#include <string>

namespace BOC{

	//declaration of ClassInfo
	class ClassInfo;
	//declaration
	class Registry;

	static std::map<std::string, ClassInfo*> mapClassInfo;

	//declaration of function to create a new instance
	typedef void* (*CreateFunction)(void* param1, void* param2, void* param3);

	//	//declaration of register function
	inline bool Register(ClassInfo* classInfo);

	class ClassInfo {
	protected:
		std::string type;
		std::string description;
		CreateFunction func;

	public:
		const std::string& GetType() const { return this->type; }
		const std::string& GetDescr() const { return this->description; }
	public:
		ClassInfo(const std::string &type, const std::string &descr, CreateFunction func) :
			type(type), description(descr), func(func){
			Register(this);
		}

		void* CreateObject(void* param1, void* param2, void* param3) const {
			return func ? (*func)(param1, param2, param3) : NULL;
		}
	};

	class Registry{
	protected:
		static std::string invalid_string;
	public:
		static bool Register(ClassInfo* classInfo){
			if (classInfo != NULL){
				if (mapClassInfo.find(classInfo->GetType()) == mapClassInfo.end()){
					mapClassInfo[classInfo->GetType()] = classInfo;
					return true;
				}
			}
			return false;
		}

		static void* CreateObject(const std::string &name, void *param1 = NULL, 
			void* param2 = NULL, void* param3 = NULL){
			std::map<std::string, ClassInfo* >::iterator iter = mapClassInfo.find(name);
			if (iter != mapClassInfo.end()){
				return ((ClassInfo*)(iter->second))->CreateObject(param1, param2, param3);
			}
			return NULL;
		}

		static const string& GetName(const std::string& name) {
			std::map<std::string, ClassInfo* >::iterator iter = mapClassInfo.find(name);
			if (iter != mapClassInfo.end()){
				return ((ClassInfo*)(iter->second))->GetType();
			}
			return invalid_string;
		}

		static const string& GetDescr(const std::string& name) {
			std::map<std::string, ClassInfo* >::iterator iter = mapClassInfo.find(name);
			if (iter != mapClassInfo.end()){
				return ((ClassInfo*)(iter->second))->GetDescr();
			}
			return invalid_string;
		}
	};

	std::string Registry::invalid_string;

	inline bool Register(ClassInfo* classInfo) {
		return Registry::Register(classInfo);
	}

#define DECLARE_CLASS \
protected: \
	static ClassInfo classInfo; \
public:\
	static void* CreateObject(void *param1, void* param2, void* param3); \
	static ClassInfo& GetClassMsg() { return classInfo; }


#define APPEND_INFO(info,name,T1,T2) \
	info.append("\n\t"); \
	info.append(name<T1,T2>::GetClassMsg().GetType()); \
	info.append(":\t"); \
	info.append(name<T1,T2>::GetClassMsg().GetDescr()); 
}

#endif
