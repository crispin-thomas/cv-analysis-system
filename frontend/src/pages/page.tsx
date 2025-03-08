import { Tabs } from "rsuite";
import { CVUpload } from "../components/cv-upload";
import { CVList } from "../components/cv-list";
import { Chat } from "../components/chat";

import "./home.css";
import { useEffect, useRef, useState } from "react";

export default function Home() {
  const contentBoxRef = useRef<HTMLDivElement>(null);
  const [chatContainerHeight, setChatContainerHeight] = useState<number>(100);

  useEffect(() => {
    const updateChatContainerHeight = () => {
      if (contentBoxRef.current) {
        const contentBoxHeight = contentBoxRef.current.offsetHeight;
        setChatContainerHeight(contentBoxHeight); // Subtract padding
      }
    };

    updateChatContainerHeight();
    window.addEventListener("resize", updateChatContainerHeight);

    return () => {
      window.removeEventListener("resize", updateChatContainerHeight);
    };
  }, []);

  console.log(chatContainerHeight)

  return (
    <div className="home-container">
      <div className="content-box" ref={contentBoxRef}>
        <h1 className="title">CV Management System</h1>
        <Tabs defaultActiveKey="query" className="custom-tabs">
          <Tabs.Tab eventKey="query" title="Query CVs">
            <Chat height={chatContainerHeight} />
          </Tabs.Tab>
          <Tabs.Tab eventKey="upload" title="Upload CV">
            <CVUpload />
          </Tabs.Tab>
          <Tabs.Tab eventKey="manage" title="Manage CVs">
            <CVList height={chatContainerHeight} />
          </Tabs.Tab>
        </Tabs>
      </div>
    </div>
  );
}
