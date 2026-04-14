
from abc import ABC
from typing import Any, Generic, List, Optional, TypeVar

from bson import ObjectId
from api.infrastructure.models.db_schemas import GeneratedImageDoc
from api.routes.models.api_schemas import QueryCondition
from api.utils.statics import default_conn_str, default_db_name, default_collection_name
import pydantic
from pymongo import MongoClient


T = TypeVar("T", bound=pydantic.BaseModel)

class MongoRepository(Generic[T], ABC):
    conn_str:str = default_conn_str
    database_name:str = default_db_name
    collection_name:str = default_collection_name
    client: MongoClient = None
    database = None
    collection = None

    def __init__(self, db_name, col_name):
        self.database_name = db_name
        self.collection_name = col_name
        self.client = MongoClient(host=self.conn_str)
        self.database = self.client[self.database_name]
        self.collection = self.database[self.collection_name]

    async def get_by_id(self, id:str) -> Optional[T]:
        entity = self.collection.find_one({"_id": ObjectId(id)})
        return None if entity is None else entity
    
    async def get(self, varname:str, value) -> Optional[T]:
        entity = self.collection.find_one({varname: value})
        return None if entity is None else entity
    
    async def get_many(self, varname:str, value) -> List[T]:
        return [doc for doc in self.collection.find({varname: value})]
    
    async def stringy_query(self, query: dict[str, Any]) -> list[T]:
        return [doc for doc in self.collection.find(query)]
    
    async def query(self, conditions: List[QueryCondition]) -> list[T]:
        query = {"$and": []}
        for condition in conditions:
            query["$and"].append({condition.field : condition.value})
            print("-- CURRENT QUERY --", query)
        return [doc for doc in self.collection.find(query)]
    
    async def get_many_containing_string(self, varname:str, match:str) -> list[T]:
        return [doc for doc in self.collection.find({varname: {'$regex': match}})]
    
    async def get_many_sorted_containing_string(self, varname:str, match:str, sortvar:str, descending:bool) -> list[T]:
        sort_dir=1
        if descending:
            sort_dir=-1
        documents = self.collection.aggregate([
            {"$match" : {varname: {'$regex': match}}},
            {"$sort" : {sortvar: sort_dir}}
        ])
        return [doc for doc in documents]
        
    async def create(self, T) -> T:
        entity = self.collection.insert_one(T.dict(by_alias=True))
        inserted = await self.get_by_id(entity.inserted_id)
        return inserted
    
    async def update(self, id:str, T) -> bool:
        update = T.dict(by_alias=True)
        del update["_id"]
        result = self.collection.update_one({"_id": ObjectId(id)}, {"$set": update})
        return result.modified_count > 0
    
    async def delete_by_id(self, id:str):
        result = self.collection.delete_one({"_id": ObjectId(id)})
        return result.deleted_count > 0
    
class GenImagesRepository(MongoRepository[GeneratedImageDoc]):
    def __init__(self, db_name:str, col_name:str = default_collection_name):
        super().__init__(db_name=db_name, col_name=col_name)